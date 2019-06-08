import logging
import os
import pickle
import time
import urllib.parse

import pandas as pd
from PIL import Image

#import pytesseract
from flask import (Flask, Response, redirect, render_template, request,
                   send_from_directory, url_for)
from functions import *
from parameters import search_word, topN
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename

# FLASK APP INIT
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# LOGGING SETUP
logging.basicConfig(filename='logfile.log', level=logging.INFO)
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.INFO)

pd.set_option('display.max_colwidth', -1)

# GLOBAL VARIABLES
processed_pdf_names = []
# Folders
UPLOAD_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/uploads/"
IMAGES_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/static/images/"
DOWNLOAD_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/download/"
TESSERACT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/Tesseract-OCR/"
#pytesseract.pytesseract.tesseract_cmd = TESSERACT_FOLDER + r"/tesseract.exe"

@app.route("/", methods=["GET", "POST"])
def start():
    return redirect("/upload")


@app.route("/upload", methods=["GET", "POST"])
def upload():

    # logging.info(f"processed pdfs : {processed_pdf_names}")

    if request.method == 'POST':
        if request.files['fileList'].filename != "":
            processed_pdf_names.clear()
            try:
                cleanup_folder(UPLOAD_FOLDER)
            except Exception as e:
                logging.debug("Folder may already be empty: {}".format(str(e)))

            logging.info("POST request received - " + time.strftime('%X %x'))
            logging.info("Processing " + str(len(request.files.getlist("fileList"))) + " files.")

            for f in request.files.getlist("fileList"):                
                if f and allowed_file(f.filename):
                    try:
                        # Saving file locally
                        filename = secure_filename(f.filename)

                        if filename.split(".")[1].lower() == "pdf":
                            f.save(UPLOAD_FOLDER+filename)                            
                        elif filename.split(".")[1].lower() == "txt":
                            f.save(UPLOAD_FOLDER+filename)

                    except Exception as e:
                        logging.error(f"Err while saving {filename}: {e}")                        
                else:
                    logging.warning("File format not allowed : " + os.path.splitext(f.filename)[1])

            return render_template("home.html", stage="pdfprocessing")

    return render_template("home.html", stage="fileselection") 


@app.route('/progress_pdfprocessing')
def progress_pdfprocessing():
    def generate():
        logging.info("Starting extraction...")
        try:
            pdf_paths = []
            pdf_images = []

            for file_name in os.listdir(UPLOAD_FOLDER):
                if file_name.endswith(".pdf"):
                    logging.info("pdf found: {}".format(file_name))
                    processed_pdf_names.append(file_name.split(".")[0])
                    file_path = UPLOAD_FOLDER + "/" + file_name
                    pdf_paths.append(file_path)
                    pdf_images.append(convert_from_path(file_path, 200))

            if len(pdf_paths) > 0:

                images_count = 0
                for pdf_image in pdf_images:
                    for image in pdf_image:
                        images_count += 1

                step = int(100/int(images_count)+1)
                progress = 0

                progress += step
                yield "data:" + str(progress) + "\n\n"

                for i in range(len(pdf_paths)):
                    logging.info(f"Extracting data from pdf : {pdf_paths[i]}")
                    pdf_name_base = os.path.splitext(os.path.basename(pdf_paths[i]))[0]
                    PageNumber = 0
                    full_text = ""

                    for image in pdf_images[i]:
                        PageNumber += 1
                        logging.info(f"Processing page {PageNumber}/{len(pdf_images[i])}...")
                        image_path = UPLOAD_FOLDER + pdf_name_base + "_" + str(PageNumber) + ".jpg"
                        image.save(image_path, "JPEG", quality=10)
                        ocr_text = pytesseract.image_to_string(Image.open(image_path), lang='eng')
                        os.remove(image_path)
                        full_text = full_text + "\n" + ocr_text.replace("|", "I")
                        progress += step
                        if progress > 100:
                            progress = 99
                        yield "data:" + str(progress) + "\n\n"

                    txt_filename = pdf_name_base + ".txt"
                    with open(UPLOAD_FOLDER + txt_filename, "w", encoding="utf-8") as f:
                        f.write(full_text)                        
            progress = 100
            yield "data:" + str(progress) + "\n\n"

        except Exception as e:
            logging.info("Error occurred while atempting to convert pdf to txt : {}".format(str(e)))
            return render_template("home.html", stage="uploadfailure")            
    return Response(generate(), mimetype='text/event-stream')


@app.route('/results_progress', methods=['GET', 'POST'])
def results_progress():
    return render_template("home.html", stage="textanalysis")


@app.route('/generate_results', methods=['GET', 'POST'])
def generate_results():
    def generate():

        try:
            start_time = time.time()

            # Loads text files to dictionary
            data = load_to_dict(UPLOAD_FOLDER, processed_pdf_names)
            yield "data:10\n\n"

            # Clean up text
            preprocessed_docs = preprocess_docs(data)
            with open(UPLOAD_FOLDER+'preprocessed_docs.data', 'wb') as filehandle:  
                # store the data as binary data stream
                pickle.dump(preprocessed_docs, filehandle) 
            yield "data:30\n\n"

            # Find most common words
            most_common_words = find_most_common_words(preprocessed_docs, topN)
            topN_keywords = [topic[0] for topic in most_common_words]
            with open(UPLOAD_FOLDER+'topN_keywords.data', 'wb') as filehandle:  
                pickle.dump(topN_keywords, filehandle) 
            yield "data:40\n\n"

            # Plot the words frequency as a bar plot to the image folder
            plot_word_freq(data, preprocessed_docs, most_common_words, IMAGES_FOLDER)
            yield "data:70\n\n"

            # Generate a word cloud image and save to the image folder
            generate_word_cloud(" ".join(preprocessed_docs), IMAGES_FOLDER)
            yield "data:80\n\n"

            # Generate data table summary with the most common words and their associated sentences
            common_words_df = most_common_words_df(data, most_common_words)
            common_words_df.to_csv(UPLOAD_FOLDER+"common_words_df.csv", sep='\t', encoding='utf-8')
            yield "data:90\n\n"

            # Generate data table summary with the given keyword (user input)
            searched_word_df = search_word_df(data, search_word)
            yield "data:100\n\n"

            # Estimate topics within documents
            #topics_df = find_topics(preprocessed_docs, num_topics=num_topics, num_words=num_words, passes=50)
            #topics_df.to_csv(UPLOAD_FOLDER+"topics_df.csv", sep='\t', encoding='utf-8')
            #yield "data:100\n\n"
            logging.info(f"Execution Time : {round(time.time() - start_time, 2)} secs")

        except Exception as e:
            logging.critical("Error occurred while generating text analysis: {}".format(str(e)))
            return redirect("/upload")

    return Response(generate(), mimetype='text/event-stream')    


@app.route('/results', methods=['GET', 'POST'])
def results():


    with open(UPLOAD_FOLDER+'topN_keywords.data', 'rb') as filehandle:  
        topN_keywords = pickle.load(filehandle)
    topN_keywords.sort()

    # list of tuples representing select options
    topic_choices = [(str(x), str(x)) for x in range(2, 7)]
    word_choices = [(str(x), str(x)) for x in range(2, 7)]
    filter_choices = [(x.title(), x.title()) for x in topN_keywords]

    if 'topics_form' in request.form:
        #print("topics form")
        with open(UPLOAD_FOLDER+'dropdown_values_topics.csv', 'w') as f:
            txt_to_write = str(request.form['topics'])+" "+str(request.form['words'])
            f.write(txt_to_write)

    elif 'filters_form' in request.form:
        #print("filters form")
        with open(UPLOAD_FOLDER+'dropdown_values_filters.csv', 'w') as f:
            f.write(request.form['filters'])
    else:
        # initialise drop down default values
        with open(UPLOAD_FOLDER+'dropdown_values_topics.csv', 'w') as f:
            text_to_write = str(request.args.get('topic_choice', '3'))+" "+str(request.args.get('word_choice', '3'))
            f.write(text_to_write)
        with open(UPLOAD_FOLDER+'dropdown_values_filters.csv', 'w') as f:
            f.write(str(request.args.get('filter_choice', filter_choices[0][0])))

    try:
        with open(UPLOAD_FOLDER+'dropdown_values_topics.csv', 'r') as f:
            txt_content = f.read().split(" ")
            topic_selection = txt_content[0]
            word_selection = txt_content[1]
        with open(UPLOAD_FOLDER+'dropdown_values_filters.csv', 'r') as f:
            filter_selection = f.read()
    except Exception as e:
        print(f"Exception when reading txt files: {e}")
   

    # application 'state' variable with default value and test
    topic_state = {'topic_choice': topic_selection}
    word_state = {'word_choice': word_selection}
    filter_state = {'filter_choice': filter_selection}

    #print("topic",topic_selection)
    #print("words",word_selection)
    #print("filter",filter_selection)

    try:
        
        with open(UPLOAD_FOLDER+'preprocessed_docs.data', 'rb') as filehandle:  
            # read the list as binary data stream
            preprocessed_docs = pickle.load(filehandle)

        # Estimate topics within documents
        topics_df = find_topics(preprocessed_docs, num_topics=int(topic_selection), num_words=int(word_selection), passes=50)
        topics_df.to_csv(UPLOAD_FOLDER+"topics_df.csv", sep='\t', encoding='utf-8')

        common_words_df = pd.read_csv(UPLOAD_FOLDER+"common_words_df.csv", sep='\t',  index_col=False, encoding="utf-8")
        common_words_df = common_words_df.drop('Unnamed: 0', 1)
        common_words_df = common_words_df[common_words_df.words == filter_selection]
        common_words_df = common_words_df.drop('words', 1)


        topics_df = pd.read_csv(UPLOAD_FOLDER+"topics_df.csv", sep='\t',  index_col=False, encoding="utf-8")
        topics_df = topics_df.drop('Unnamed: 0', 1)


        return render_template(
            "home.html",
            stage="results",
            topN=topN,
            doc_list=get_file_names_string(UPLOAD_FOLDER, processed_pdf_names),
            common_words_df_column_names=common_words_df.columns,
            common_words_df_row_data=list(common_words_df.values.tolist()),
            zip=zip,
            topics_df_column_names=topics_df.columns,
            topics_df_row_data=list(topics_df.values.tolist()),
            topic_choices=topic_choices,
            word_choices=word_choices,
            filter_choices=filter_choices,
            topic_state=topic_state,
            word_state=word_state,
            filter_state=filter_state) 

    except Exception as e:
        print("Error occurred while rendering result page: {}".format(str(e)))
        return redirect("/upload")

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download_sample(filename):
    try:
        return send_from_directory(directory=DOWNLOAD_FOLDER, filename=filename, as_attachment=True)
    except Exception as e:
        logging.critical("Exception when downloading file: {}".format(str(e)))
        return render_template("upload.html", stage="fileselection")



# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == "__main__":
    app.run(debug=False)
