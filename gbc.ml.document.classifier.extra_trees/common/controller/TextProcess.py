import ast
import os
import pickle
import sys
from queue import Queue
from threading import Thread

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from common.model.ClassFile import ClassFile
from common.model.Document import Document
from common.model.Sentence import Sentence
from common.model.Singleton import Singleton
from common.nlp.NlpStopwords import NlpStopwords
from common.nlp.NlpUtils import NlpUtils
from commonsLib import loggerElk


class TextProcess:

    def __init__(self, conf, nlp):
        self.conf = conf
        self.nlp = nlp
        self.tf = None
        self.tf_idf = None
        self.encoder = None
        self.logger = loggerElk(__name__, True)

    def process(self, text, kind='none', path=''):
        """
        process texts

        """
        xdoc = Document()
        xdoc.kind = kind
        xdoc.path = path

        self.logger.Information(f"NLP: Cleaning text {path}...")

        doc = None
        sentences = []
        if len(text) > self.conf.max_string_size:
            self.logger.Information(len(text))
            split = NlpUtils.split_by_size(text, self.conf.max_string_size)
            for t in split:
                doc = self.nlp(t)
                for s in doc.sents:
                    sentences.append(s)
        else:
            doc = self.nlp(text)
            _ = doc.sents

        # mark stopwords
        for sentence in doc.sents:
            # self.logger.Information(sentence)
            s = Sentence()
            for token in sentence:
                t = NlpStopwords.clean_token(self.conf, token)
                s.add_token(t)

            xdoc.add_sentence(s)

        # build 1-grams (just lemmas)
        for s in xdoc.sentences:
            for t in s.tokens:
                if not t.stop:
                    xdoc.add_gram(t.lemma.lower())

        # self.logger.Information(xdoc.path, ':\n', ' '.join(sorted(list(xdoc.grams))))
        ClassFile.list_to_file(list(xdoc.grams), ClassFile.file_base_name(xdoc.path) + '.gram')
        # self.logger.Information('', ' '.join(files.file_to_list(files.file_base_name(xdoc.path) + '.gram')))
        return xdoc

    def load_vector_models(self):
        self.tf = ClassFile.load_model(os.path.join(self.conf.working_path, self.conf.tf))
        self.tf_idf = ClassFile.load_model(os.path.join(self.conf.working_path, self.conf.tfidf))
        # self.logger.Information(self.tf.vocabulary_)

    def load_vectorizer_model(self):
        self.encoder = Singleton.getInstance(self.conf).getFirstEncoder()
        if self.encoder is None:
            self.logger.Error("Encoder file not found!", sys.exc_info())

    def load_vectorizer_model_s3(self, s3_files, s3_service):
        # Obtener fichero tfidf y hacer
        self.encoder = Singleton.getInstance(self.conf).getFirstEncoderS3(s3_files, s3_service)
        if self.encoder is None:
            self.logger.Error("Encoder file not found!", sys.exc_info())

    def get_tfidf(self, gram):
        count = self.tf.transform([' '.join(gram)])
        vector = self.tf_idf.transform(count)
        return vector

    def get_tfidf_from_vectorizer(self, gram):
        if self.encoder is None:
            self.load_vectorizer_model()
        if self.encoder is not None:
            vector = self.encoder.transform([' '.join(gram)])
            return vector
        return None

    def get_tfidf_from_vectorizer_s3(self, s3_files, gram_to_process, s3_service):
        if self.encoder is None:
            self.load_vectorizer_model_s3(s3_files, s3_service)
        if self.encoder is not None:
            vector = self.encoder.transform([' '.join(gram_to_process)])
            return vector
        return None

    def get_count(self, gram):
        count = self.tf.transform([' '.join(gram)])
        # self.logger.Information(count.shape)
        return count

    def transform(self, data='', file=''):
        if self.encoder is None:
            self.load_vectorizer_model()
        text = ClassFile.get_text(os.path.join(self.conf.working_path, data, file))
        return self.transform_text(text, file)

    def transform_data(self, data=''):
        if self.encoder is None:
            self.load_vectorizer_model()
        file_text_list = ClassFile.list_files_ext(self.conf.working_path, "txt")
        file_text_list_filter = ClassFile.filter_by_size(file_text_list)
        text_list = list()
        for file in file_text_list_filter:
            file_data = ClassFile.get_text(os.path.join(self.conf.working_path, data, file))
            text_list.extend(self.transform_text(file_data, file))
        return text_list

    def transform_data_s3(self, tfidf_file, s3_service, file_data=''):
        if self.encoder is None:
            self.load_vectorizer_model_s3(tfidf_file, s3_service)
        doc = self.process(NlpUtils.clean_text(file_data), 'none')
        vector = self.get_tfidf_from_vectorizer_s3(tfidf_file, doc.grams, s3_service)
        if vector is not None:
            return vector
        return None

    def transform_text(self, text, path):
        doc = self.process(NlpUtils.clean_text(text), 'none', path=path)
        # self.logger.Information(doc.grams)
        vector = self.get_tfidf_from_vectorizer(doc.grams)
        text_list = list()
        if vector is not None:
            text_list.append(vector)
        return text_list

    def _do_pre_process(self, q, result):  # q:[[index, text, kind, path], ...]
        """
        launch text processing in threads

        """
        while not q.empty():
            work = q.get()  # fetch new work from the Queue
            try:
                self.logger.Information("Requested..." + str(work[0]))
                data = self.process(work[1], work[2], work[3])
                result[work[0]] = data  # Store data back at correct index
                self.logger.Information("Done..." + str(work[0]))
            except Exception as e:
                result[work[0]] = Document()
                self.logger.Error('TextProcess::do_pre_process::{}'.format(e), sys.exc_info())

            # signal to the queue that task has been processed
            q.task_done()
        return True

    def _create_dataset(self, docs):
        """
        build the tf and tfidf matrixes for the whole text

        """
        text = []
        for doc in docs:
            text.append(doc.get_grams_as_text())

        # create the transform
        # tokenize and build vocab
        count_vectorizer = CountVectorizer()
        x_tf = count_vectorizer.fit_transform(text)
        self.logger.Information(x_tf.shape)

        # idf
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_tf)
        self.logger.Information(x_tfidf.shape)

        # encode documents
        for doc in docs:
            vector_tf = count_vectorizer.transform([doc.get_grams_as_text()])
            self.logger.Information(vector_tf.shape)
            self.logger.Information(type(vector_tf))
            self.logger.Information(vector_tf.toarray())
            ClassFile.save_sparse_csr(ClassFile.file_base_name(doc.path) + '.tf', vector_tf)

            vector_tfidf = tfidf_transformer.transform(vector_tf)
            self.logger.Information(vector_tfidf.shape)
            self.logger.Information(type(vector_tfidf))
            self.logger.Information(vector_tfidf.toarray())
            ClassFile.save_sparse_csr(ClassFile.file_base_name(doc.path) + '.tfidf', vector_tfidf)

        return x_tf, x_tfidf

    def create_dataset_from_unigrams_direct(self, uni_grams, local_storage=True):
        text = []
        for doc_grams in uni_grams:
            if len(doc_grams) == 0:
                self.logger.Information('.< size 0 vector >.')
            else:
                text.append(' '.join(list(doc_grams)))

        # create the transform
        vectorizer = TfidfVectorizer(min_df=1, max_df=0.99)
        x_tfidf = vectorizer.fit_transform(text)
        self.logger.Information('tfidf shape:', x_tfidf.shape)
        if local_storage:
            vector_file_path = os.path.join(self.conf.working_path, self.conf.vectorizer)
            ClassFile.save_model(vector_file_path, vectorizer)
            self.logger.Information('vectorizer saved', vector_file_path)

        return vectorizer

    def _create_dataset_from_uni_grams(self, uni_grams):
        """
        Build the tf and tfidf matrixes for the whole text loading all .gram files

        """
        text = []
        for doc_grams in uni_grams:
            if len(doc_grams) == 0:
                self.logger.Information('.< size 0 vector >.')
            else:
                text.append(' '.join(list(doc_grams)))

        # create the transform
        # tokenize and build vocab
        count_vectorizer = CountVectorizer()
        x_tf = count_vectorizer.fit_transform(text)
        self.logger.Information(x_tf.shape)

        # idf
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_tf)
        self.logger.Information(x_tfidf.shape)

        ClassFile.save_model(os.path.join(self.conf.working_path, self.conf.tf), count_vectorizer)
        ClassFile.save_model(os.path.join(self.conf.working_path, self.conf.tfidf), tfidf_transformer)

        return x_tf, x_tfidf

    def pre_process_file(self, file):
        """
        Process txt file to get the .grams

        """
        gram_item = os.path.join(self.conf.working_path, file)
        gram_item_filter = ClassFile.filter_by_size(gram_item)

        data = None
        if gram_item_filter and len(gram_item_filter) > 0:
            f = gram_item_filter.pop()
            category = ClassFile.get_containing_dir_name(f)
            text = NlpUtils.clean_text(ClassFile.get_text(f))
            data = self.process(text, category, f)
            self.logger.Information(f"Document {file} was pre-processed")

        return data

    def pre_process_batches(self):
        """
        Process all txt files in batches to get the .grams

        """
        categories = set()
        all_categories = []

        gram_list = ClassFile.list_files_ext(self.conf.working_path, ".txt")
        gram_list_filter = ClassFile.filter_by_size(gram_list)
        no_gram_list_filter = ClassFile.filter_gram_duplicate(
            self.conf.working_path, gram_list_filter)
        print(f"Processing: {len(no_gram_list_filter)} GRAMs")
        all_docs = [None for _ in no_gram_list_filter]
        q = Queue(maxsize=0)

        counter = 0
        total = len(no_gram_list_filter)
        i = 0

        while i < total:
            h = i
            for _ in range(self.conf.pre_process_batch_size):
                if h < total:
                    f = no_gram_list_filter[h]
                    category = ClassFile.get_containing_dir_name(f)
                    categories.add(category)

                    all_categories.append(category)
                    text = NlpUtils.clean_text(ClassFile.get_text(f))

                    self.logger.Information('doc %s to q' % (counter + 1))
                    q.put((counter, text, category, f))

                    counter += 1
                h += 1

            for _ in range(q.qsize()):
                worker = Thread(target=self._do_pre_process, args=(q, all_docs))
                worker.setDaemon(True)  # setting threads as "daemon" allows main program to
                # exit eventually even if these dont finish
                # correctly.
                worker.start()

            # now we wait until the queue has been processed
            q.join()

            q.empty()
            i = h

        self.logger.Information(len(categories), categories)
        # create_dataset(conf, all_docs)

    def pre_process_batches_s3(self, s3_service, all_docs):
        """
        Process all txt files in batches to get the .grams

        """
        categories = set()
        all_categories = []
        gram_list = []
        for doc in all_docs:
            if doc.Size > 0 and doc.Key.upper().endswith('.TXT'):
                gram_list.append(doc)
        # gram_list = ClassFile.list_files_ext(self.conf.working_path, self.conf.text_file_ext)
        # gram_list_filter = ClassFile.filter_by_size(gram_list)

        # TODO Cuando se graben los gram, ver si ya existe antes de tratarlo y quitarlo de gram_list
        # no_gram_list_filter = ClassFile.filter_gram_duplicate(self.conf.working_path, gram_list_filter)
        no_gram_list_filter = gram_list

        all_docs = [None for _ in no_gram_list_filter]
        q = Queue(maxsize=0)

        counter = 0
        total = len(no_gram_list_filter)
        i = 0

        while i < total:
            h = i
            for j in range(self.conf.pre_process_batch_size):
                if h < total:
                    f = no_gram_list_filter[h]
                    spl = f.Key.split("/")
                    if len(spl) >= 2:
                        category = spl[len(spl) - 2]
                    elif len(spl) == 1:
                        return spl[0]
                    else:
                        raise Exception(f"The element with key {f.Key} is not in a folder for category")
                    categories.add(category)

                    all_categories.append(category)

                    # Obtain txt

                    text = s3_service.get_txt_file(f.Key)

                    text = NlpUtils.clean_text(text)

                    self.logger.Information('doc %s to q' % (counter + 1))
                    q.put((counter, text, category, f.Key, s3_service))

                    counter += 1
                h += 1

            for j in range(q.qsize()):
                worker = Thread(target=self._do_pre_process, args=(q, all_docs))
                worker.setDaemon(True)  # setting threads as "daemon" allows main program to
                # exit eventually even if these dont finish
                # correctly.
                worker.start()

            # now we wait until the queue has been processed
            q.join()

            q.empty()
            i = h

        self.logger.Information(len(categories), categories)
        # create_dataset(conf, all_docs)

    def create_full_dataset_vectorizer(self):
        """
        Load all .gram files and call create_dataset_from_unigrams

        """
        v_list = ClassFile.list_files_ext(self.conf.working_path, ".gram")
        unigrams = []
        self.logger.Information(v_list)

        for f in v_list:
            unigrams.append(ClassFile.file_to_list(f))

        self.create_dataset_from_unigrams_direct(unigrams)

    def create_full_dataset_vectorizer_s3(self, s3_service):
        """
        Load all .gram files from s3 and call create_dataset_from_unigrams

        """
        v_list = s3_service.get_files_from_s3()
        v_list = [x for x in v_list if x.Key.upper().endswith(".GRAM")]
        unigrams = []
        self.logger.Information(v_list)

        for f in v_list:
            s3Content = s3_service.get_txt_file(f.Key)

            unigrams.append(ast.literal_eval(s3Content))
        vectorizer = self.create_dataset_from_unigrams_direct(unigrams, False)
        pickle_byte_obj = pickle.dumps(vectorizer)
        path = s3_service.domain + "/" + self.conf.vectorizer
        s3_service.upload_file(path, pickle_byte_obj)
        print("Success")

    def _pre_process(self):
        """
        Process all txt files to get the .grams

        """
        categories = set()
        all_categories = []

        d_list = ClassFile.list_files_ext(self.conf.working_path, "txt")
        all_docs = [None for _ in d_list]
        q = Queue(maxsize=0)

        counter = 0
        total = len(d_list)
        for f in d_list:
            category = ClassFile.get_containing_dir_name(f)
            categories.add(category)

            all_categories.append(category)
            text = NlpUtils.clean_text(ClassFile.get_text(f))

            self.logger.Information('doc %s to q' % (counter + 1))
            q.put((counter, text, category, f))

            counter += 1

        for _ in range(total):
            worker = Thread(target=self._do_pre_process, args=(q, all_docs))
            worker.setDaemon(True)  # setting threads as "daemon" allows main program to
            # exit eventually even if these dont finish
            # correctly.
            worker.start()
        # now we wait until the queue has been processed
        q.join()

        self.logger.Information(len(categories), categories)
        # create_dataset(conf, all_docs)
