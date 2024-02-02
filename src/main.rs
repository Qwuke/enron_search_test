use std::cmp::Ordering;
use std::io::{self, Read};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use lazy_static::lazy_static;
use bigdecimal::{FromPrimitive, BigDecimal};
use qp_trie::{wrapper::BString, Trie};

lazy_static! {
    static ref PUNCTUATION_CHARS: HashSet<String> = vec!["!", "\"", "#", "$", "%", 
        "&", "'", "(", ")", "*", "+", ",", ";", ".", "/", ":", ",", "<", "=",
        ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "-"]
        .iter()
        .map(|s| s.to_string())
        .collect::<HashSet<String>>();
}

fn main() {
    let mut args: Vec<String> = env::args().collect();

    let search_term = args.pop().expect("to have single argument");
    if args.len() != 1 {
        println!("Please use a single argument");
        return;
    }
    
    println!("Searching for {}", search_term);

    let email_file_paths = get_email_paths_from_dir("/home/qwuke/enron_search_engine/resources/enron/").expect("Cannot read files");

    let document_word_freq = get_document_word_freq(email_file_paths); 
    
    let inverse_document_frequency = calc_inverse_document_freq(document_word_freq.clone());

    let tf_idf = calc_tf_idf(document_word_freq, inverse_document_frequency);

    let mut search_trie = build_search_trie(tf_idf);
    
    search(&search_term, &mut search_trie);
}

fn search(input: &str, search_trie: &mut Trie<BString, BTreeMap<BigDecimal, String>>) {
    let sanitized_input = input
        .to_lowercase()
        .chars()
        .filter(|&ch| !PUNCTUATION_CHARS.contains(&ch.to_string()))
        .collect::<String>();

    let mut matched_prefixes: Vec<(&BString, String, BigDecimal)> = search_trie.iter_prefix_mut(&BString::from(sanitized_input.clone()))
        .flat_map(|(word, map)| {
            let mut temp_vec = Vec::new();
            for _i in 1..10 {
                if let Some((score, doc)) =  map.pop_last() {
                    temp_vec.push((word, doc.to_owned(), score.to_owned()));
                };
            }
            temp_vec
        })
        .collect();

    matched_prefixes.sort_by(|(word1, _, score1), (word2, _, score2)| { 
        if word1.as_str().eq(&sanitized_input) && word2.as_str().ne(&sanitized_input)  {
            return Ordering::Greater;
        } else if word1.as_str().ne(&sanitized_input) && word2.as_str().eq(&sanitized_input) {
            return Ordering::Less;
        }
        score1.cmp(score2)
    });
    matched_prefixes.reverse();

    if matched_prefixes.is_empty() {
        println!("No matches");
    } else {
        matched_prefixes.iter().take(100)
            .for_each(|(word, doc, score)|{
                println!("Email {} matching word {} with score {}", doc, word.as_str(), score);
            });
    }
}

fn build_search_trie(tf_idf: HashMap<String, HashMap<String, f64>>) -> Trie<BString, BTreeMap<BigDecimal, String>> {
    let words_with_ranked_docs = tf_idf
        .into_iter()
        .fold(HashMap::new(), 
            |mut acc, (doc, word_score_map)| {
                word_score_map.into_iter()
                    .for_each(|(word, score)| {
                        let ordered_score = BigDecimal::from_f64(score).expect("Valid real decimal");
                        acc.entry(word)
                            .and_modify(|map: &mut BTreeMap<BigDecimal, String>| { map.insert(ordered_score.clone(), doc.clone()); })
                            .or_insert({ 
                                let mut new_map = BTreeMap::new();
                                new_map.insert(ordered_score, doc.clone());
                                new_map 
                            });
                    });
                acc });


    let mut search_trie = Trie::new();
    words_with_ranked_docs
        .into_iter()
        .for_each(|(word, doc_rankings)| {
            search_trie.insert(BString::from(word), doc_rankings);
        });

    search_trie
}

fn calc_tf_idf(document_word_freq: HashMap<String, HashMap<String, u64>>, inverse_document_frequency: HashMap<String, f64>) -> HashMap<String, HashMap<String, f64>> {
    let document_tf_score = document_word_freq.into_iter()
        .map(|(document, word_freq)| {
            let total_word_count = word_freq.values().sum::<u64>(); 
            let term_frequency = word_freq
                .into_iter()
                .map(|(word, doc_word_count)| 
                    (word.to_owned(), 
                    doc_word_count as f64 / total_word_count as f64))
                .collect::<HashMap<String, f64>>();
            (document, term_frequency)
        })
        .collect::<HashMap<String, HashMap<String, f64>>>();

    let doc_tf_idf = document_tf_score.iter()
        .map(|(document, tf_scores)| {
            let tf_idf_scores = tf_scores.into_iter()
                .map(|(word, tf_score)| {
                    (word.to_owned(), tf_score * inverse_document_frequency.get(word).unwrap_or(&0.0_f64))
                })
            .collect::<HashMap<String, f64>>();
            
            let normalized_tf_idf = l2_normalize(tf_idf_scores);

            (document.to_string(), normalized_tf_idf)
        }) 
        .collect::<HashMap<String, HashMap<String, f64>>>();

    doc_tf_idf
}

fn l2_normalize(tf_id: HashMap<String, f64>) -> HashMap<String, f64> {
    let l2_norm = tf_id
        .values()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();

    tf_id
        .iter()
        .map(|(key, value)| (key.clone(), value / l2_norm))
        .collect::<HashMap<String, f64>>()
}

fn calc_inverse_document_freq(document_word_freq: HashMap<String, HashMap<String, u64>>) -> HashMap<String, f64> {
    let total_word_freq = document_word_freq
        .values()
        .fold( HashMap::new(), 
            |mut acc, doc_map| {
                doc_map.into_iter()
                    .for_each(|(k, v)| {
                        acc.entry(k).and_modify(|v| *v += 1).or_insert(1);
                    });
                acc });
    
    let total_document_count = document_word_freq.keys().len();

    let inverse_document_frequency = total_word_freq
        .into_iter()
        .map(|(word, count)| {
            let documents_with_term = (total_document_count as f64 + 1.0) / (count as f64 + 1.0);
            (word.to_owned(), documents_with_term.ln() + 1.0)
        })
        .collect::<HashMap<String, f64>>();
    inverse_document_frequency
}

fn get_document_word_freq(email_file_paths: Vec<PathBuf>) -> HashMap<String, HashMap<String, u64>> {
    email_file_paths.iter()
            .map(|file_path| {
                let file_name = file_path.clone().as_os_str().to_str()
                    .expect("OS string path contained invalid valid UTF8").to_owned();
                
                let mut file = File::open(file_path).expect("File could not be opened from path");
                let mut buf = vec![];
                file.read_to_end(&mut buf).expect("File could not be read into byte buffer");
                
                // Removes not UTF8 characters from emails
                let file_content = String::from_utf8_lossy (&buf).into_owned();

                let words_in_file = file_content
                    .split_whitespace()
                    .collect::<Vec<&str>>();
                let mut word_count: HashMap<String, u64> = HashMap::new();
                
                for word in words_in_file.iter() {
                    let sanitized_word = word
                        .to_lowercase()
                        .chars()
                        .filter(|ch| !PUNCTUATION_CHARS.contains(&ch.to_string()))
                        .collect::<String>();
                    word_count.entry(sanitized_word).and_modify(|count| *count += 1).or_insert(1);
                }
            
                (file_name, word_count)
            })
            .collect::<HashMap<String, HashMap<String, u64>>>()
}


fn get_email_paths_from_dir(path: impl AsRef<Path>) -> std::io::Result<Vec<PathBuf>> {
    let mut buf = vec![];
    let entries = fs::read_dir(path)?;

    for entry in entries {
        let entry = entry?;
        let meta = entry.metadata()?;

        if meta.is_dir() {
            let mut subdir = get_email_paths_from_dir(entry.path())?;
            buf.append(&mut subdir);
        }

        if meta.is_file() {
            buf.push(entry.path());
        }
    }

    Ok(buf)
}