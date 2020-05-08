import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import random
import glob
import csv

from sklearn.svm import SVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from numpy import array_split
from config import DevConfig
import preprocessing
import Model_final
import batch_model


UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['csv'])

app=Flask(__name__)
app.config.from_object(DevConfig)
app.config['UPLOAD_FOLDER'] =  os.path.abspath(UPLOAD_FOLDER)
app.secret_key = 'Active Learning'
classifiers = {
            'random_forest':[RandomForestClassifier, 0],
            'svc':[SVC, 1],
            'knn':[KNeighborsClassifier, 0],
            'nb':[GaussianNB,0]
        }


@app.route('/')
def index():
    return render_template('main.html')

@app.route('/front_page')
def front_page():
    return render_template('front_page.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/load_dataset', methods=['GET', 'POST'])
def load_dataset():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'training_dataset' not in request.files\
                or 'test_dataset' not in request.files\
                or 'unlabel_dataset' not in request.files:
            flash('Please upload all three files')
            return redirect(request.url)
        files = dict((
            ('td_file', request.files['training_dataset']),
            ('test_file', request.files['test_dataset']),
            ('ud_file', request.files['unlabel_dataset'])
        ))

        for uploaded_file in files:
            if files[uploaded_file].filename == '':
                flash('No file data for {}'.format(uploaded_file))
                return redirect(request.url)

        for uploaded_file in files:
            if files[uploaded_file] and allowed_file(files[uploaded_file].filename):
                filename = uploaded_file+'.csv'
                files[uploaded_file].save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                flash('Unable to save file {}'.format(uploaded_file))
                return redirect(request.url)
        return redirect('/data_preprocessing')
    if request.method == 'GET':
        return render_template('load_dataset.html')

@app.route('/data_preprocessing')
def preprocess():
    data_sentences = {
        'td_file':[],
        'test_file':[],
        'ud_file':[]
    }
    data_frame = dict()
    for data_source in data_sentences.keys():
        temp = preprocessing.extract_sentences(os.path.abspath('./uploads/'+data_source+'.csv'))
        data_sentences[data_source] = temp['sentences']
        data_frame[data_source] = temp['data_frame']

    all_sentences = data_sentences['td_file'] + data_sentences['test_file'] + data_sentences['ud_file']
    model = preprocessing.generate_model(all_sentences)

    w2vsentences = dict()
    for data_source in ['td_file', 'test_file']:
        w2vsentences[data_source] = preprocessing.w2v_sentence(data_sentences[data_source], data_frame[data_source], model, data_source+'_outfile')

    w2vsentences['ud_file'] = preprocessing.w2v_unlabel(data_sentences['ud_file'], data_frame['ud_file'], model, 'ud_file_outfile')

    shapes = [data_frame['td_file'].shape,data_frame['test_file'].shape,data_frame['ud_file'].shape]
    data_lengths = [len(data_sentences['td_file']), len(data_sentences['test_file']), len(data_sentences['ud_file']), len(all_sentences)]
    w2v_lengths = [len(w2vsentences['td_file']), len(w2vsentences['test_file']), len(w2vsentences['ud_file'])]
    w2v_first = [w2vsentences['td_file'][0], w2vsentences['test_file'][0], w2vsentences['ud_file'][0]]


    return render_template('preprocessing.html',df_shapes=shapes, data_lengths=data_lengths, w2v_lengths=w2v_lengths, w2v_first=w2v_first)

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'GET':
        return render_template('process.html')
    if request.method == 'POST':
        classifier = request.form['classifier']
        return redirect(url_for('process_questions', classifier=classifier))
@app.route('/process/questions', methods=['GET','POST'])
def process_questions():
    # Getting the data from the pickled files
    unlabel_data = Model_final.get_data('ud_file_outfile')
    train_data = Model_final.get_data('td_file_outfile')
    test_data = Model_final.get_data('test_file_outfile')

    # Initialize the data
    unlabel_X = Model_final.initialise_unlabel_array(unlabel_data)
    train_X = Model_final.initialise_training_array(train_data)['X']
    train_Y = Model_final.initialise_training_array(train_data)['Y']
    test_X = Model_final.initialise_training_array(test_data)['X']
    test_Y = Model_final.initialise_training_array(test_data)['Y']

    if request.method == 'GET':
        classifier = classifiers[request.args.get('classifier')][0]()
        has_decision_function = classifiers[request.args.get('classifier')][1]
        if request.args.get('classifier') == 'svc':
            classifier = svm.SVC(probability=True)
            classifier.fit(train_X, train_Y)
        if request.args.get('classifier')=='random_forest':
            classifier=RandomForestClassifier(n_estimators=1000,oob_score=True)
            classifier.fit(train_X,train_Y)
        question_ids = Model_final.get_random_questions( decision_function=has_decision_function,
                                                     unlabel_X=unlabel_X,
                                                     unlabel_data=unlabel_data,
                                                     num_questions=5,
                                                     clf=classifier
                                                     )
        questions = [ unlabel_data[n]['Actual sentence'] for n in question_ids ]

        return render_template('questions.html',
                               question_numbers = range(len(questions)),
                               questions = questions,
                               question_ids = question_ids,
                               classifier = request.args.get('classifier')
                               )

    if request.method == 'POST':
        if request.form == None:
            flash('No marked questions')
            return redirect('/process')

        training_output = []
        new_shapes = []
        # Get data from the request
        classifier = classifiers[request.form.get('classifier')][0]()
        has_decision_function = classifiers[request.form.get('classifier')][1]
        if request.form.get('classifier') == 'svc':
            classifier = svm.SVC(probability=True)
            classifier.fit(train_X, train_Y)
        elif request.form.get('classifier')=='random_forest':
            classifier=RandomForestClassifier(n_estimators=100,oob_score=True)
            classifier.fit(train_X,train_Y)
        elif request.form.get('classifier')=='knn':
            classifier=KNeighborsClassifier(n_neighbors=10)
            classifier.fit(train_X,train_Y)
        elif request.form.get('classifier')=='nb':
            classifier=GaussianNB()
            classifier.fit(train_X,train_Y)

        train_count = Model_final.count_hate_speech(train_data)
        test_count = Model_final.count_hate_speech(test_data)
        p_train = float(train_count)/len(train_data) * 100
        p_test = float(test_count)/len(test_data) * 100

        #For saving purposes
        map(os.unlink, glob.glob('/static/*.csv'))

        labeled_file = open('static/labeled_sentences.csv', 'w')
        writer = csv.writer(labeled_file)
        writer.writerows([["index","label", "text"]])

        # prepare the full list of selections
        selections_list = []
        indexes = request.form.get('question_ids')[1:-1]
        for sentence_id in indexes.split(', '):
            if request.form.get(sentence_id, 'off') == 'on':
                selections_list.append({'index': int(sentence_id), 'label': 1})
                writer.writerows([[unlabel_data[int(sentence_id)]["index"],'1',unlabel_data[int(sentence_id)]["Actual sentence"]]])
                #writer.writerows([['1',unlabel_data[int(sentence_id)]["Actual sentence"]]])
            else:
                selections_list.append({'index': int(sentence_id), 'label': 0})
                writer.writerows([[unlabel_data[int(sentence_id)]["index"],'0',unlabel_data[int(sentence_id)]["Actual sentence"]]])
                #writer.writerows([['0',unlabel_data[int(sentence_id)]["Actual sentence"]]])

        # Running the initial training
        training_output.append( Model_final.apply_training( unlabel_data, train_X, train_Y, test_X, test_Y, classifier, './static/Results_unlabel.csv') )
        n=1
        from operator import itemgetter
        newlist = sorted(selections_list, key=itemgetter('index'), reverse=True)
        for sublist in array_split(newlist, 10):
            temp = Model_final.evaluate(sublist, unlabel_data, train_X, train_Y, unlabel_X)
            train_X, train_Y, unlabel_data, unlabel_X = temp['train_X'], temp['train_Y'], temp['unlabel_data'], temp['unlabel_X']
            new_shapes.append ({'training_new_shape': train_X.shape, 'unlabel_new_length': len(unlabel_data)})
            training_output.append( Model_final.apply_training(unlabel_data, train_X, train_Y, test_X, test_Y, classifier, './static/Results_unlabel-{}.csv'.format(n)) )
            n+=1

        first_training_output = training_output[0]
        results = zip(training_output[1:], new_shapes)

        return render_template('questions.html', train_count=train_count, p_train=p_train, test_count=test_count, p_test=p_test,
                               first_training_output = first_training_output, results = results
                               )

@app.route('/process_batch_mode', methods=['GET', 'POST'])
def process_batch_mode():
    if request.method == 'GET':
        return render_template('process_batch_mode.html')
    if request.method == 'POST':
        classifier = request.form['classifier']
        question_num=request.form['question_num']
        return redirect(url_for('process_questions_batch_mode', classifier=classifier,question_num=question_num))

@app.route('/process/questions_batch_mode', methods=['GET','POST'])
def process_questions_batch_mode():
    # Getting the data from the pickled files
    unlabel_data = batch_model.get_data('ud_file_outfile')
    train_data = batch_model.get_data('td_file_outfile')
    test_data = batch_model.get_data('test_file_outfile')
    question_num=request.args.get('question_num')
    # Initialize the data
    unlabel_X = batch_model.initialise_unlabel_array(unlabel_data)
    train_X = batch_model.initialise_training_array(train_data)['X']
    train_Y = batch_model.initialise_training_array(train_data)['Y']
    test_X = batch_model.initialise_training_array(test_data)['X']
    test_Y = batch_model.initialise_training_array(test_data)['Y']

    if request.method == 'GET':
        classifier = classifiers[request.args.get('classifier')][0]()
        has_decision_function = classifiers[request.args.get('classifier')][1]

        if request.args.get('classifier') == 'svc':
            classifier.fit(train_X, train_Y)
        question_ids = batch_model.get_random_questions(decision_function=has_decision_function,
                                                        unlabel_X=unlabel_X,
                                                        unlabel_data=unlabel_data,
                                                        num_questions=5,
                                                        clf=classifier
                                                        )
        questions = [unlabel_data[n]['Actual sentence'] for n in question_ids]

        return render_template('question_batch_mode.html',
                               question_numbers=range(len(questions)),
                               question_num=question_num,
                               questions=questions,
                               question_ids=question_ids,
                               classifier=request.args.get('classifier')
                               )
        #for testing purpose
        #return render_template('test_batch.html',
        #                       question_numbers=range(len(questions)),
        #                       question_num=question_num,
        #                       questions=questions,
        #                       question_ids=question_ids,
        #                       classifier=request.args.get('classifier')
        #                       )
    if request.method == 'POST':
        if request.form == None:
            flash('No marked questions')
            return redirect('/process')
        map(os.unlink, glob.glob('/static/*.csv'))


        training_output = []
        new_shapes = []
        # Get data from the request
        classifier = classifiers[request.form.get('classifier')][0]()
        has_decision_function = classifiers[request.form.get('classifier')][1]
        if request.form.get('classifier') == 'svc':
            classifier.fit(train_X, train_Y)

        train_count = batch_model.count_hate_speech(train_data)
        test_count = batch_model.count_hate_speech(test_data)
        p_train = float(train_count)/len(train_data) * 100
        p_test = float(test_count)/len(test_data) * 100

        # prepare the full list of selections
        selections_list = []
        group_selection=[]
        indexes = request.form.get('question_ids')[1:-1]
        questions=request.form.get('questions')


        for sentence_id in indexes.split(', '):
            if request.form.get(sentence_id, 'off') == 'on':
                selections_list.append({'index': int(sentence_id), 'label': 1})
                group_selection.append({'index':int(sentence_id),'label':1})
            else:
                selections_list.append({'index': int(sentence_id), 'label': 0})
                group_selection.append({'index':int(sentence_id),'label':0})
        another_list=[]
        question_index = []
        k=0
        j=5
        for i in range(0,20):

           question_index.append(selections_list[k])
           another_list.append(selections_list[k:j])
           k=k+5
           j=j+5

        selection_len=selections_list[0:5]

        for elem in another_list:
            sync = False
            for i in elem:
                if i['label'] == 1:
                    sync = True
                    break
            if sync:
                for i in elem:
                    i['label'] = 1
        map(os.unlink, glob.glob('/static/*.csv'))

        labeled_file = open('static/labeled_sentences.csv', 'w')
        writer = csv.writer(labeled_file)
        save_questions=[]
        for elem in selections_list:
            writer.writerows([[elem['index'],unlabel_data[int(elem['index'])]["Actual sentence"],elem['label']]])
            #save_questions.append(unlabel_data[int(elem['index'])]["Actual sentence"])

        labeled_file.close()
        #comment from here
        # Running the initial training
        training_output.append( batch_model.apply_training( unlabel_data, train_X, train_Y, test_X, test_Y, classifier, './static/Results_unlabel.csv') )
        n=1
        from operator import itemgetter
        #newlist = sorted(selections_list, key=itemgetter('index'), reverse=True)
        newlist = sorted(selections_list, key=itemgetter('index'), reverse=True)
        for sublist in array_split(newlist, 10):
            temp = batch_model.evaluate(sublist, unlabel_data, train_X, train_Y, unlabel_X)
            train_X, train_Y, unlabel_data, unlabel_X = temp['train_X'], temp['train_Y'], temp['unlabel_data'], temp['unlabel_X']
            new_shapes.append ({'training_new_shape': train_X.shape, 'unlabel_new_length': len(unlabel_data)})
            training_output.append( batch_model.apply_training(unlabel_data, train_X, train_Y, test_X, test_Y, classifier, './static/Results_unlabel-{}.csv'.format(n)) )
            n+=1

        first_training_output = training_output[0]
        results = zip(training_output[1:], new_shapes)

        return render_template('question_batch_mode.html', train_count=train_count, p_train=p_train, test_count=test_count, p_test=p_test,
                               first_training_output = first_training_output, results = results
                               )

        #to here
        #return render_template('test_batch.html', train_count=0, p_train=0,
        #                       test_count=0, p_test=0,
        #                       first_training_output =0,
        #                       #results = 0,
        #                       questions_index=another_list,slen=selection_len
        #

         #                      )

    return render_template('question_batch_mode.html')
if __name__=='__main__':
    app.run()