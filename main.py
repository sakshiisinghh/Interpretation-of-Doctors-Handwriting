import argparse
import json
from typing import Tuple, List

import cv2
import editdistance
from path import Path
import csv

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor
import enchant

import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image

from io import StringIO 
import os 
import keras.utils as ku

class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'

def get_img_height() -> int:
    """Fixed height for NN."""
    return 32

def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)

def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())

def train(model: Model,
          loader: DataLoaderIAM,
          line_mode: bool,
          early_stopping: int = 25) -> None:
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_accuracies = []
    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
    best_char_error_rate = float('inf')  # best validation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

        # validate
        char_error_rate, word_accuracy = validate(model, loader, line_mode)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        write_summary(summary_char_error_rates, summary_word_accuracies)

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped.')
            break

def validate(model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy

words = {} #similar and medicine
condition = []
mylist = []
medicine = "medicine"

def infer(model: Model, fn_img: Path) -> None:
    global mylist
    global condition 
    global words
    global medicine

    #"""Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    #print(f'Recognized: "{recognized[0]}"')
    #print(f'Probability: {probability[0]}')
    count = 0
    recognizedword = str(recognized[0]).capitalize()
    
    with open(r'C:/Users/trvee/Documents/Semester 8/drugsComTest_raw.csv(1)/DrugDataset.csv') as file_obj:
        reader_obj = csv.reader(file_obj)
        for row in reader_obj:
            count = count+1
            string1 = str(row[0]).capitalize() 
            similar = (enchant.utils.levenshtein(string1,recognizedword))
            #if similar == 0 or similar == 1 or similar == 2 or similar == 3 or similar == 4 or similar == 5:
                #if(string1[0]==recognizedword[0]):
            words[similar] = string1
    mylist = list(words.keys())
    #print("Medicine : " ,words[min(mylist)])
    with open(r'C:/Users/trvee/Documents/Semester 8/drugsComTest_raw.csv(1)/DrugDataset.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            if str(row[0]).capitalize() == str(words[min(mylist)]):
                condition.append(row[1])
                break
    #print("Condition :" ) 
    #for i in set(condition):
        #print(i, end=" ")
    medicine = str(words[min(mylist)])

##Streamlit 
st.set_page_config(page_title="Handwriting Recognition", page_icon=":tada:", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Use local CSS
#def local_css(file_name):
#    with open(file_name) as f:
#        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#local_css("../style/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_tutvdkg0.json")

# ---- SPACE ----
padding_top = 0
padding_bottom = 0
st.markdown(f"""<style>
        .reportview-container .main .block-container{{
            padding-top: {padding_top}rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)

# ---- HEADER SECTION ----
with st.container():
    c1, c2 = st.columns([1,5])
    with c1:
        st_lottie(lottie_coding, height=200, key="coding")
    with c2:
        st.markdown("<h1 style='padding-top:70px;font-size: 32pt;'>Know Your Medicine...</h1>",unsafe_allow_html=True)
        
#---- WHAT I DO ----
#with st.container():
st.write("---")
background_color = "pink"
left_column, centre, right_column = st.columns([3,0.5,2])
with left_column:
    padding_top = 0
    padding_bottom = 0
    st.markdown(f"""<style>
            .reportview-container .main .block-container{{padding-top: {padding_top}rem;}}</style>""",unsafe_allow_html=True,)
    st.markdown("<h2>Upload An Image</h2>", unsafe_allow_html=True)
    image_file = st.file_uploader("",type=['png'])
    if image_file is not None:
        #file_details = {"FileName":image_file.name,"FileType":image_file.type}
        #st.write(str(file_details))
        img = ku.load_img(image_file)
        st.image(img,width=550)
        with open(os.path.join(r'C:/Users/trvee/Documents/Semester 8/Major Project/SimpleHTR-master/SimpleHTR-master - Copy/data',image_file.name),"wb") as f: 
            f.write(image_file.getbuffer())         
        st.success("Saved File")
        nameoffile = str(image_file.name)
    else:
        nameoffile = 'blank.png'
   
def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default=os.path.join('../data/',nameoffile))
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()


#def main():
    """Main function."""
 
    # parse arguments and set CTC decoder
args = parse_args()
decoder_mapping = {'bestpath': DecoderType.BestPath,
                    'beamsearch': DecoderType.BeamSearch,
                    'wordbeamsearch': DecoderType.WordBeamSearch}
decoder_type = decoder_mapping[args.decoder]

# train the model
if args.mode == 'train':
    loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)

    # when in line mode, take care to have a whitespace in the char list
    char_list = loader.char_list
    if args.line_mode and ' ' not in char_list:
        char_list = [' '] + char_list

    # save characters and words
    with open(FilePaths.fn_char_list, 'w') as f:
        f.write(''.join(char_list))

    with open(FilePaths.fn_corpus, 'w') as f:
        f.write(' '.join(loader.train_words + loader.validation_words))

    model = Model(char_list, decoder_type)
    train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)

# evaluate it on the validation set
elif args.mode == 'validate':
    loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
    model = Model(char_list_from_file(), decoder_type, must_restore=True)
    validate(model, loader, args.line_mode)

# infer text on test image
elif args.mode == 'infer':
    model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
    infer(model, args.img_file)

with right_column:
    if nameoffile != "blank.png":
        #print("list : ",medicine)
        st.markdown("<h1 style = background-color:lightblue;</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='height:100px;'>Results</h2>", unsafe_allow_html=True)
        st.markdown("<h3><u>Medicine</u></h3>", unsafe_allow_html=True)
        st.subheader(medicine)
        st.markdown("<h3 style='padding-top:50px'><u>Condition</u></h3>", unsafe_allow_html=True)
        for i in set(condition):
            st.subheader(i)
        st.markdown("<h1 style = background-color:lightblue;</h1>", unsafe_allow_html=True)

#if __name__ == '__main__':
#    main()