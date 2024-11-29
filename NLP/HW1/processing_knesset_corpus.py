import argparse
from docx import Document
import os
import re
import json
import pandas as pd

def hebrew_to_number(hebrew_text):
    hebrew_numerals = {
        "אחד": 1, "שתיים": 2, "שלוש": 3, "ארבע": 4, "חמש": 5,
        "שש": 6, "שבע": 7, "שמונה": 8, "תשע": 9, "עשר": 10,
        "עשרים": 20, "שלושים": 30, "ארבעים": 40, "חמישים": 50,
        "שישים": 60, "שבעים": 70, "שמונים": 80, "תשעים": 90,
        "מאה": 100, "מאתיים": 200
    }

    suffixes = {
        "מאות": 100, "עשרה" : 10
    }

    total = 0
    parts = re.split(r'-', hebrew_text)
    # by now we should have an array of the hebrew numbers/suffixes

    for i, part in enumerate(parts):
        part = part.strip()
        
        if part.startswith('ו'):
            part = part[1:]

        if part in hebrew_numerals:
            if i + 1 < len(parts) and parts[i+1] in suffixes:
                if parts[i+1] == "מאות":
                    total += hebrew_numerals[part] * suffixes[parts[i+1]]
                elif parts[i+1] == "עשרה":
                    total += hebrew_numerals[part] + suffixes[parts[i+1]]
                continue
            else:
                total += hebrew_numerals[part]
                
    return total

def parse_protocol_metadata(file_path):
    filename = file_path.split('/')[-1]

    ## knesset num ##
    knesset_num = filename.split('_')[0]
    knesset_num = knesset_num.split('\\')[-1]

    ## type ##
    if filename.split('_')[1][2] == 'm':
        protocol_type = 'plenary'
    elif filename.split('_')[1][2] == 'v':
        protocol_type = 'committee'

    ## protocol number ##
    protocol_number = -1

    document = Document(file_path)
    lines = [para.text for para in document.paragraphs]

    for line in lines[:500]:
        if "פרוטוקול מס'" in line:
            match = re.search(r"פרוטוקול מס'\s*(\d+)", line)
            if match:
                protocol_number = match.group(1)
        if re.search(r"הישיבה ה.*? של הכנסת ה.*", line):
            match = re.search(r"הישיבה ה(.*?) של", line)           # get the hebrew number only
            hebrew_number = match.group(1).strip()                  # extract the text
            protocol_number = hebrew_to_number(hebrew_number)       # use a function to determine the number
    
    return knesset_num, protocol_type, protocol_number

def clean_speaker_name(raw_name):
    prefixes_to_remove = [
        r'היו"ר', r'יו"ר', r'ח"כ', r'ד"ר', r'פרופ\'', r'עו"ד', 
        r'נצ"מ', r'ניצב',
        r'מר', r'גברת',
        r'שר', r'שרת', r'השר', r'השרה',
        r'סגן', r'סגנית',
        r'הבינוי והשיכון',
        r'העבודה הרווחה והשירותים החברתיים',
        r'האוצר',
        r'התעשייה והמסחר',
        r'החינוך',
        r'מזכיר המדינה', r'מזכירת המדינה',
        r'מזכיר הכנסת', r'מזכירת הכנסת',
        r'התחבורה',
        r'הפנים',
        r'המשפטים',
        r'והתרבות',
        r'הכלכלה והתכנון',
        r'לביטחון פנים',
        r'לביטחון',
        r'הבריאות',
        r'דובר_המשך',
        r'העבודה, הרווחה והשירותים החברתיים',
        r'התשתיות הלאומיות, האנרגיה והמים',
        r'לאיכות הסביבה',
        r', התרבות והספורט',
        r'העבודה והרווחה',
        r'תשובת', r'תשובה',
        r'התקשורת',
        r'במשרד ראש הממשלה',
        r'המדע, התרבות והספורט',
        r'התיירות',
        r'התעשייה, המסחר והתעסוקה',
        r'נשיא הפרלמנט האירופי',
        r'התשתיות הלאומיות',
        r'החקלאות ופיתוח הכפר',
        r'המדע והטכנולוגיה',
        r'לקליטת העלייה',
        r'התשתיות',
        r'מ"מ',
        r'ועדת העבודה, הרווחה והבריאות',
        r'והבטיחות בדרכים',
        r'ראש הממשלה',
        r'המשנה לראש הממשלה',
        r'הביטחון',
        r'להגנת הסביבה',
        r'הכנסת',
        r'לנושאים אסטרטגיים ולענייני מודיעין',
        r'לאזרחים ותיקים',
        r'המודיעין',
        r'הכלכלה',
        r'החקלאות'      
    ]

    # prefix patterns to remove
    prefix_pattern = r'\b(?:' + r'|'.join(prefixes_to_remove) + r')\b'

    cleaned_name = re.sub(r'\(.*?\)', '', raw_name)
    cleaned_name = re.sub(prefix_pattern, '', cleaned_name)
    cleaned_name = re.sub(r'<.*?>', '', cleaned_name)
    cleaned_name = cleaned_name.replace("<","").replace(">","")
    return cleaned_name.strip()

def extract_sentences(file_path):
    data = []
    found_first_speaker = False

    doc = Document(file_path)
    speaker = None
    speech = []

    for para in doc.paragraphs:
        if(not found_first_speaker):    # discussion has not begun
            # check if the current line is underlined
            try:
                is_underlined = (
                    para.style.font.underline or 
                    (para.style.base_style and para.style.base_style.font.underline) or 
                    (para.runs and para.runs[0].underline)
                )
            except AttributeError:
                is_underlined = False
            
            # if the current line is underlined, has "היו"ר" and a colon, 
            if(is_underlined and 
               (re.search(r'היו"ר .*:', para.text) or
                re.search(r'היו”ר .*:', para.text) or
                re.search(r'יו"ר .*:', para.text) or
                re.search(r'יו”ר .*:', para.text) or
                # This case occurs once in the files so we used this specific case
                ('16_ptv_577758' in file_path and re.search(r'שלמי גולדברג .*:', para.text)))
               ):
                match = re.match(r'(.*?):\s*(.*)', para.text)
                if match:
                    speaker = match.group(1).strip()
                    speaker = clean_speaker_name(speaker)
                    speech = [re.sub(r'<<.*?>>', '', match.group(2)).strip()]
                    if speech != ['']:
                        data.append({"speaker_name": speaker, "sentence_text": " ".join(speech)})
                found_first_speaker = True
                continue
        else:                           # discussion has begun
            if(para.alignment == 1):    # 1:centered  
                continue                # skip centered paragraphs (titles, votes etc...)
            
            if "הישיבה ננעלה בשעה" in para.text:
                break
            # check if the current line is underlined
            try:
                is_underlined = (
                    para.style.font.underline or 
                    (para.style.base_style and para.style.base_style.font.underline) or 
                    (para.runs and para.runs[0].underline)
                )
            except AttributeError:
                is_underlined = False

            if ":" in para.text and is_underlined:      # potential speaker
                match = re.match(r'(.*?):\s*(.*)', para.text)
                if match:
                    if speaker and speech != ['']:
                        data.append({"speaker_name": speaker, "sentence_text": " ".join(speech)})
                        speech = []
                    
                    # update the speaker and speech
                    speaker = match.group(1).strip()
                    speaker = clean_speaker_name(speaker)
                    if speech != ['']:
                        speech.append(re.sub(r'<<.*?>>', '', match.group(2)).strip())
            elif speaker and speech:
                text = para.text.replace(">","")
                if text:
                    speech.append(text.strip())
    
    if speaker:
        data.append({"speaker_name": speaker, "sentence_text": " ".join(speech)})
    
    return data

def tokenize_sentence(sentence):
    pattern = r'[א-ת]+|\d+|[.,!?;:\-\']'
    return re.findall(pattern, sentence)

def produce_corpus(dir_path, write_to_txt = False):
    df = pd.DataFrame(columns=[ "protocol_name",
                                "knesset_number", 
                                "protocol_type", 
                                "protocol_number", 
                                "speaker_name", 
                                "sentence_text"
                                ])
    docx_count = 0
    docx_processed = 0

    for _, _, files in os.walk(dir_path):
        for file in files:    
            if file.endswith('.docx'):
                docx_count += 1

    print(f"{docx_processed}/{docx_count} docx files processed.", end='\r')
    
    # extracting speeches
    for curr_file in os.listdir(dir_path):
        if curr_file.endswith('.docx'):
            file_path = os.path.join(dir_path, os.fsdecode(curr_file))
            knesset_num, protocol_type, protocol_num = parse_protocol_metadata(file_path)
            data = extract_sentences(file_path)

            for entry in data:
                entry["protocol_name"] = curr_file
                entry["knesset_number"] = knesset_num
                entry["protocol_type"] = protocol_type
                entry["protocol_number"] = protocol_num
            
            temp_df = pd.DataFrame(data)

            # sentences tokenization
            expanded_data = []
            for _, row in temp_df.iterrows():
                speech = row["sentence_text"]
                sentences = re.split(r'[.!?]', speech)

                full_sentences = [
                    sentences[i].strip() + sentences[i + 1]
                    for i in range(0, len(sentences) - 1, 2)
                ]

                
                for sentence in full_sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if "– – –" in sentence or "– –" in sentence:
                        continue
                    if re.search(r'[a-zA-Z]', sentence):                      #exclude sentences with numbers
                        continue
                    tokens = tokenize_sentence(sentence)
                    
                    if len(tokens) >= 4:
                        new_entry = row.to_dict()
                        new_entry["sentence_text"] = sentence
                        expanded_data.append(new_entry)

            expanded_df = pd.DataFrame(expanded_data)
            df = pd.concat([df, expanded_df], ignore_index=True)

            # DEBUG PURPOSES - write full texts to txt files
            if write_to_txt:
                output_file = os.path.splitext(curr_file)[0] + ".txt"
                output_path = os.path.join("output", output_file)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"Knesset Number: {knesset_num}\n")
                    f.write(f"Type: {protocol_type}\n")
                    f.write(f"Protocol Number: {protocol_num}\n\n")
                    for entry in data:
                        f.write(f"Speaker: {entry['speaker_name']}\n")
                        f.write(f"Speech: {entry['sentence_text']}\n")
                        f.write("\n")
            
            print(f"{docx_processed}/{docx_count} docx files processed.", end='\r', flush = True)
            docx_processed += 1
    
    print(f"All docx files processed.", flush = True)
    return df

def save_dataframe_to_jsonl(df, output_path):
    with open(output_path, 'w', encoding='utf-8') as jsonl_file:
        for _, row in df.iterrows():
            json_object = row.to_dict()
            jsonl_file.write(f"{json.dumps(json_object, ensure_ascii=False)}\n")

# to run the script in the command line, use the following syntax:
# 'python processing_knesset_corpus.py <path/to/protocols> <path/to/output/file>'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory containing protocol files."
    )

    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the directory for the output JSONL."
    )
    
    args = parser.parse_args()
    
    dir_path = args.directory
    out_path = args.output_path

    print("Processing Protocols...")
    df = produce_corpus(dir_path)

    # write all names
    with open("Speakers.txt", "w", encoding="utf-8") as f:
        for name in df['speaker_name'].unique():
            f.write(f"{name}\n")

    save_dataframe_to_jsonl(df, out_path)
    print(f"Corpus was saved to {out_path}.")

if __name__ == "__main__":
    main()