import csv
import itertools
import os
import collections
import pathlib
import re
import pandas as pd
import matplotlib.pyplot as plot
from difflib import SequenceMatcher
from lexical_diversity import lex_div as ld
from nltk import word_tokenize
from nltk.util import bigrams
import statistics
import string

# Farben für Output
red = '\033[31m'
green = '\033[32m'
yellow = '\033[33m'
reset = '\033[0m'


def string_to_drama_format(path_to_directory):
    '''
    String_to_drama_format
    :param path_to_directory: path to the directory where the project is stored
    :return: empty
    A function that takes a directory, loops through all .txt files in the drama format
    and formats the drama in such a way that it is sentence tokenized and the number of persons in the drama are counted
    Author: Charlotte
    '''
    num_persons_all_dramas = []
    directory = os.fsencode(path_to_directory + "/dramas")

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        output_path = os.path.join(path_to_directory, "dramas", "output_format")

        if filename.endswith(".txt"):

            path_filename = os.path.join(path_to_directory, "dramas", filename)
            output_path_metadaten = os.path.join(output_path, "metadata", filename)
            output_path = os.path.join(output_path, filename)
            with open(path_filename, mode='r') as f:
                drama = f.read()
                drama_formatted, num_persons = sent_tokenize(drama)
                num_persons_all_dramas.append([filename, num_persons, drama_formatted])
    return num_persons_all_dramas


def sent_tokenize(drama):
    '''
    format_drama
    :param drama: a drama that has been read in with f.read as one single string
    :return: all sentences in the drama as a list, and the number of speakers in the drama
    this function uses a regex to create multi-name-entities into one single name in order to split the sentence by
    whitespace ('FRAU HOLLE'->'FRAUHOLLE'). It is a weird workaround but it works.
    It goes through all words in the tokenized drama, and creates a list of all sentences with the original writing of
    the speaker in the beginning of the sentence ('FRAU HOLLE').

    Finally, the method also returns the number of speakers, which it takes from counting the number of entities in
    the names dictionary
    Author: Charlotte
    '''
    # write multi names without space before splitting
    pattern = r'\b([A-Z]+(?: [A-Z]+)*)\b'
    regex = re.compile(pattern)
    # create a dictionary to store the names with and without whitespace in between
    names = {}
    drama, count = regex.subn(lambda match: match.group(1).replace(" ", ""), drama)
    # zip with and without whitespace names
    names.update(zip(regex.findall(drama), regex.findall(drama)))
    # split drama into words by whitespace
    drama_words = drama.split()
    index = 0
    cur_sent = ""
    all_sent_list = []
    for word in drama_words:
        # strip word from punctuation (like in 'PRINCESS:')
        word_no_punct = re.sub(r'[^\w]', '', word)
        # initial word is always PRINZESSIN/a speaker
        if index == 0:
            cur_sent = word
        else:
            if word_no_punct.isupper():
                # replace name by the version with whitespace in it
                for key, value in names.items():
                    word = re.sub(value, key, word)
                # append finished sentence to the list
                all_sent_list.append(cur_sent)
                # reset cur_sent string
                cur_sent = ""
            # add current word to cur_sent
            cur_sent = cur_sent + " " + word
        index += 1
    # append last sentence
    all_sent_list.append(cur_sent)
    num_persons = len(names)
    return all_sent_list, num_persons


def all_dramas_pd(all_dramas):
    '''
    All_dramas_pd
    :param: all_dramas: a list of all dramas
    :return: a dataframe containing all information of all_dramas, converted to a pandas dataframe, with headers
    Author: Charlotte
    '''
    # convert to a pandas dataframe
    persons_df = pd.DataFrame(all_dramas)
    persons_df.columns = ['filename', 'num_persons', 'drama']
    return persons_df


# by @heidekrauttt
def mean_sentence_length(dramas_df):
    '''
    mean_sentence_length
    A function that calculates the mean sentence length of all dramas based on the 'drama' column of the dataframe
    :param: dramas_df: a dataframe containing all dramas as a pandas dataframe
    :return: the dataframe with an additional column with the added feature mean sentence length
    Author: Charlotte
    '''
    mean_sent_lengths = []
    for index in range(0, len(dramas_df)):
        sent_lengths = []
        text = dramas_df.iloc[index]['drama']
        if len(text) > 1:
            for sentence in text:
                if len(sentence) != 1:
                    sentence_list = strip_speaker_from_sentence(sentence)
                    if len(sentence_list) == 2:
                        speech_part = sentence_list[1]
                        speech_part = speech_part.split()
                        sent_lengths.append(len(speech_part))
            mean = sum(sent_lengths) / len(sent_lengths)
            mean_sent_lengths.append(mean)
        else:
            mean_sent_lengths.append(0)

    dramas_df['mean_sentence_length'] = mean_sent_lengths
    return dramas_df


# by @TheynT
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# by @TheynT
def similar_sentence(dramas_df):
    '''
    similar_sentence

    :param dramas_df: takes a dataframe
    :return dramas_df: returns the input df with a new column mean_sentence_similarity

    Iterates over every sentence in a scene and compares it to every other sentence.
    It then calculcates the mean similarity between every sentence.
    The mean similarity of every scene is then fed into a df.

    It also calculates the mean similarity between all processed scenes. (Only output in debug)
    If the debug var is set to True it outputs additional information during runtime.
    Author: Leon
    '''
    mean_gens = []
    debug = False
    for index in range(0, len(dramas_df)):
        simlist = []
        clean_sentence_list = []
        text = dramas_df.iloc[index]['drama']
        if debug:
            print(f"{yellow}{dramas_df.iloc[index]['filename']}{reset}")
            print(text)
        for sentence in text:
            if len(sentence) != 1:
                sentence_list = strip_speaker_from_sentence(sentence)
                if len(sentence_list) == 2:
                    clean_sentence_list.append(sentence_list[1])
        if debug:
            print(f"{clean_sentence_list}")
        clean_sentence_list = list(filter(None, clean_sentence_list)) # filters faulty "" strings
        if len(clean_sentence_list) == 0 or len(clean_sentence_list) == 1:
            simlist.append(1)
            if debug:
                print(f"{yellow}Scene only has one Speaker.{reset}")
        for i in range(len(clean_sentence_list)):
            for j in range(i + 1, len(clean_sentence_list)):
                sim = similar(clean_sentence_list[i], clean_sentence_list[j])
                if debug:
                    print(f"{yellow}Ähnlichkeit zwischen{reset}{clean_sentence_list[i]}{yellow} und{reset}{clean_sentence_list[j]}{yellow} ist: {sim}{reset}")
                simlist.append(sim)
        mean = sum(simlist) / len(simlist)
        if debug:
            if mean >= 0.5:          # alles über 0.5 ist zu ähnlich
                print(f"{yellow}Mean similarity of sentences between Speakers:{reset} {red}{round(mean, 4)}{reset}")
            elif 0.33 <= mean < 0.5: # alles zwischen 0.33 und 0.5 ist fast zu ähnlich
                print(f"{yellow}Mean similarity of sentences between Speakers:{reset} {yellow}{round(mean, 4)}{reset}")
            elif 0.1 <= mean < 0.33: # alles zwischen 0.1 und 0.33 ist unterschiedlich genug
                print(f"{yellow}Mean similarity of sentences between Speakers:{reset} {green}{round(mean, 4)}{reset}")
            elif mean < 0.1:         # alles unter 0.1 ist zu unterschiedlich
                print(f"{yellow}Mean similarity of sentences between Speakers:{reset} {red}{round(mean, 4)}{reset}")
        mean_gens.append(mean)
    mean_total = sum(mean_gens) / len(mean_gens)
    if debug:
        print(f"{round(mean_total, 4)}")
    dramas_df['mean_sentence_similarity'] = mean_gens
    return dramas_df


# by @TheynT & @Mel
def phrasenwiederholung_lexicalrichness(dramas_df):
    '''
    phrasenwiederholung

    :param dramas_df: takes a dataframe
    :return dramas_df: returns the input df with a new column scene_has_phrase_repetition

    Iterates over every drama in dramas_df, every sentence in a drama.
    Forms bigrams for every sentence then checks if those bigrams occur >= considered_phrase_repetition (default: 3).
    If a repetition is found, counts it and saves every repetition of a scene in a list.
    Then adds list as a new colum to dramas_df, before returning it.

    If the debug var is set to True it outputs additional information during runtime.

    Perkuhn, Rainer, Holger Keibel & Marc Kupietz. 2012. Korpuslinguistik. Paderborn: Fink.
    Authors: Leon, Melina
    '''
    # change this var to alter how many occurances of the same bigram are considered repetitive
    # default 3 to tolerate a single repetition (e.g. "Oh ja, Oh ja! Die Sauce, sie schmeckt.")
    considered_phrase_repetition = 3
    # change this var to alter how many occurances of the same bigram are considered repetitive
    debug = False
    allwords = []
    ttr_list = []
    moving_ttr_list = []
    mtld_list = []
    repetition_list = []
    for index in range(0, len(dramas_df)):
        text = dramas_df.iloc[index]['drama']
        if debug:
            print(f"{yellow}{dramas_df.iloc[index]['filename']}{reset}")
            print(text)
        repetition_counter = 0 # reset for every new drama
        for sentence in text:
            split_sentence = strip_speaker_from_sentence(sentence)
            if len(split_sentence) >= 2:
                sentence = split_sentence[1]
            words = word_tokenize(sentence) # tokenize sentence with nltk
            for w in words: # filters all punctuations
                if w in string.punctuation:
                    words.remove(w)
            allwords.extend(words)
            bigram_list = list(bigrams(words))
            bigram_count = collections.Counter(bigram_list)
            for bigram, count in sorted(bigram_count.items()):      
                if count >= considered_phrase_repetition:
                    if debug:
                        print(f'{red}"%s" is repeated %d time%s.{reset}' % (bigram, count, "s" if count > 1 else ""))
                    repetition_counter = repetition_counter + 1
        if debug:
            print(f"TTR: {ld.ttr(allwords)}")
            print(f"Moving Average TTR: {ld.mattr(allwords)}")
            print(f"MTLD: {ld.mtld(allwords)}")
        ttr_list.append(ld.ttr(allwords))
        moving_ttr_list.append(ld.root_ttr(allwords))
        mtld_list.append(ld.mtld(allwords))
        allwords.clear()
        repetition_list.append(repetition_counter)
    dramas_df['ttr'] = ttr_list
    dramas_df['moving_ttr'] = moving_ttr_list
    dramas_df['mtld'] = mtld_list
    dramas_df['repetitive_ngrams'] = repetition_list
    return dramas_df


# by @juliwer
def speech_length_variation(dramas_df):
    '''
    (heavily based on mean_sentence_length method)
    Measures variation of speech length for each drama and saves it to the dataframe.
    Adds three columns to the dataframe:
        speech_lengths: list of the length of each speech part in a generation (measured in tokens)
        speech_length_span: span between shortest and longest speech part in a generation
        speech_length_stdev: standard deviation as measure of variance in speech length per drama
    Returns modified Dataframe.
    Author: Julia
    '''
    # create new column for saving absolute lengths of each speech part (as lists)
    speech_lenghts_in_drama = []
    # create new column for saving length span for each drama to
    speech_length_span_in_drama = []
    # create new columns for saving other variation measures
    speech_length_standard_derivation = []
    # loop over dramas in dataframe
    for index in range(0, len(dramas_df)):

        text = dramas_df.iloc[index]['drama']
        # initialize list to save length of each speech part in text to
        speech_lengths = []

        for sentence in text:
            if len(sentence) != 1:
                sentence_list = strip_speaker_from_sentence(sentence)
                if len(sentence_list) == 2:
                    speech_part = sentence_list[1]
                    # tokenize speech part
                    speech_part = speech_part.split()
                    # save speech length of each sentence to speech length list
                    speech_lengths.append(len(speech_part))

        # save lengths for each drama to speech_leghts_in_drama (append)
        speech_lenghts_in_drama.append(speech_lengths)
        # check if speech length is not empty
        # save length span for each drama
        if len(speech_lengths) > 0:
            max_lenght, min_length = max(speech_lengths), min(speech_lengths)
            speech_length_span_in_drama.append(max_lenght - min_length)
        else:
            speech_length_span_in_drama.append(None)

        if len(speech_lengths) > 1:
            # TODO calculate standard derivation for each drama
            standard_dev = statistics.stdev(speech_lengths)
            speech_length_standard_derivation.append(standard_dev)
        else:
            speech_length_standard_derivation.append(None)

    # add columns to dataframe
    dramas_df['speech_lengths'] = speech_lenghts_in_drama
    dramas_df['speech_length_span'] = speech_length_span_in_drama
    dramas_df['speech_length_stdev'] = speech_length_standard_derivation
    return dramas_df


def strip_speaker_from_sentence(sentence):
    '''
    strip_speaker_from_sentence
    Helper method
    :param: sentence: one sentence
    :return: sentence with split speaker (identified as everything before the first ":")
    Author: Charlotte
    '''
    sentence = sentence.split(':', 1)
    return sentence


def plot_mean_persons(persons_df):
    '''
    plot_mean_persons: plots the mean amount of persons over all dramas
    :param persons_df: dataframe with the rows ['filename', 'num_persons', 'drama']
    :return: nothing, draws a plot
    Author: Charlotte
    '''
    persons_df["num_persons"].plot(kind='hist')
    plot.savefig('mean_num_persons.png')
    return


def plot_mean_sentences(mean_sentence_df):
    '''
    plots the mean amount of persons over all dramas
    :param persons_df: dataframe with the rows ['filename', 'num_persons', 'drama']
    :return: nothing, draws a plot
    Author: Charlotte
    '''
    mean_sentence_df["mean_sentence_length"].plot(kind='hist')
    plot.savefig('mean_sentence_length_hist.png')
    mean_sentence_df["mean_sentence_length"].plot()
    plot.savefig('mean_sentence_length.png')
    return


# by @juliwer
def plot_column_of_df(df: pd.DataFrame, col: str, filename: str):
    '''
    creates histogram of data from one column of a dataframe
    :param:
        df: DataFrame of dramas
        col: string indicating the name of the column we want to plot
        filename: string indicating the name of the file to which we want
        to save the results
    :returns: saves histogram to separate file
    Author: Julia
    '''
    plot.figure()
    df[col].plot(kind='hist')
    plot.savefig(filename)
    return


# by @heidekrauttt
def check_string_occurrences(drama_strings):
    '''
    check_string_occurrences
    Helper method for occurrences(): Checks if same speaker occurs twice directly after one another
    :param: drama_strings: list of one drama separated into sentence strings
    :return: True if double occurrence, else false
             numeric value of number of speakers
    Author: Charlotte
    '''
    uppercase_strings = []
    for sentence in drama_strings:
        for word in sentence.split(" "):
            if word.isupper() and word.endswith(":"):
                uppercase_strings.append(word)

    unique_uppercase_strings = set(uppercase_strings)
    # check if same speaker occurs twice directly after one another
    previous_string = None
    for current_string in uppercase_strings:
        if current_string == previous_string:
            return True, len(unique_uppercase_strings)
        previous_string = current_string
    return False, len(unique_uppercase_strings)


# by @heidekrauttt
def occurrences(dramas_df):
    '''
    occurrences
    Checks if same speaker occurs twice directly after one another
    :param: dramas_df: pandas dataframe containing all dramas with their data
    :return: the dataframe with one added column containing a boolean (that is 1 if double occurrence, else 0)
    Author: Charlotte
    '''
    for index in range(0, len(dramas_df)):
        text = dramas_df.iloc[index]['drama']
        occurrence, num_speakers = check_string_occurrences(text)
        if num_speakers != dramas_df.iloc[index]['num_persons']:
            dramas_df.at[index, 'num_persons'] = num_speakers
        dramas_df.at[index, 'double_occurrence'] = int(occurrence)
    return dramas_df


# by @heidekrauttt and @juliwer
def food_words(dramas_df, cwd):
    '''
    food_words
    Counts how many food words we find in each drama (verbs and nouns) by use of two corpora
    :param: dramas_df: pandas dataframe containing all dramas with their data
            cwd: for file I/O
    :return: the dataframe with one added column for the number of food nouns and one for number of food verbs
    Author: Charlotte and Julia
    '''
    # get food verbs from corpus
    cwd1 = cwd + '/foodverbs_corpus.txt'
    with open(cwd1, 'r', encoding='utf-8') as f:

        verblist = f.read()
        verblist = verblist.split('\n')
   
    # get food nouns from corpus
    cwd2 = cwd + '/foodnouns_corpus.txt'
    with open(cwd2, 'r', encoding='utf-8') as f:
        nounlist = f.read()
        nounlist = nounlist.split('\n')[:-1]

    for index in range(0, len(dramas_df)):
        food_verb_count, food_noun_count = 0, 0
        text = dramas_df.iloc[index]['drama']
        for sentence in text:
            for verb in verblist:
                if verb + ' ' in sentence:  # assuming there are spaces behind verbs
                    food_verb_count += 1
            for noun in nounlist:
                if noun + ' ' in sentence:
                    food_noun_count += 1
        dramas_df.at[index, 'food_verb_count'] = int(food_verb_count)
        dramas_df.at[index, 'food_noun_count'] = int(food_noun_count)
    return dramas_df


# by @heidekrauttt
def raw_text(dramas_df):
    '''
    raw_text
    Adds the raw text of each drama as a parameter. Is only really used in main method to calculate number of words
    in one drama, but might be nice to have.
    :param: dramas_df: pandas dataframe containing all dramas with their data
    :return: the dataframe with one added column containing one string with all the text in it
    Author: Charlotte
    '''
    for index in range(0, len(dramas_df)):
        raw_text = ""
        text = dramas_df.iloc[index]['drama']
        for sentence in text:
            raw_text = raw_text + str(sentence)
        dramas_df.at[index, 'raw_text'] = raw_text
    return dramas_df


def best_generations(dramas_df, path, minimal_mean_sentence_length = 10, occurrences = 0, min_num_speakers = 0,
                    max_num_speakers = 4, min_humor_ann_1 = 0, max_humor_ann_1 = 3, min_humor_ann_2 = 0,
                    max_humor_ann_2 = 3, min_speech_length_span = 0, max_speech_length_span = 100,
                    min_speech_length_stdev = 0, max_speech_length_stdev = 100, min_food_verbs = 0, min_food_nouns = 0,
                     min_regression_score = 0.0, trial = 0):
    '''
        best_generations
        Filters the dataframe according to input parameters. This is the core of the evaluation tool.
        The results are written into a file at path location
        :param:
            dramas_df:                      pandas dataframe containing all dramas with their data
            path:                           path to write results to (.csv format)
            minimal_mean_sentence_length:   minimal mean sentence length for one drama
            occurrences:                    if a drama can have a double occurrence (same speaker twice)
            min_num_speakers:               minimal number of speakers in a drama
            max_num_speakers:               maximal number of speakers in a drama
            min_humor_ann_1:                minimal humor annotation (Annotator 1)
            max_humor_ann_1:                maximal humor annotation (Annotator 1)
            min_humor_ann_2:                minimal humor annotation (Annotator 2)
            max_humor_ann_2:                maximal humor annotation (Annotator 2)
            min_speech_length_span:         minimal speech length span for one drama
            max_speech_length_span:         maximal speech length span for one drama
            min_speech_length_stdev:        minimal speech length standard deviation for one drama
            max_speech_length_stdev:        maximal speech length standard deviation for one drama
            min_food_verbs:                 minimal number of food verbs in one drama
            min_food_nouns:                 minimal number of food nouns in one drama
            min_regression_score:           minimal regression score if available
            trial:                          the trial we are in. If number is not 1, the header line will not be written
                                            to the output file.
        :return: empty
        Author: Charlotte
        '''

    output_file_name = path
    if 'Annotation_1' in dramas_df.columns and 'Annotation_2' in dramas_df.columns:
        filtered_dramas_df = dramas_df[(dramas_df['mean_sentence_length'] > minimal_mean_sentence_length) &
                                       (dramas_df['double_occurrence'] == occurrences) &
                                       (dramas_df['num_persons'] > min_num_speakers) &
                                       (dramas_df['num_persons'] < max_num_speakers) &
                                       (dramas_df['Annotation_1'] > min_humor_ann_1) &
                                       (dramas_df['Annotation_1'] < max_humor_ann_1) &
                                       (dramas_df['Annotation_2'] > min_humor_ann_2) &
                                       (dramas_df['Annotation_2'] < max_humor_ann_2) &
                                       (dramas_df['speech_length_span'] > min_speech_length_span) &
                                       (dramas_df['speech_length_span'] < max_speech_length_span) &
                                       (dramas_df['speech_length_stdev'] > min_speech_length_stdev) &
                                       (dramas_df['speech_length_stdev'] < max_speech_length_stdev) &
                                       (dramas_df['food_verb_count'] > min_food_verbs) &
                                       (dramas_df['food_noun_count'] > min_food_verbs)
        ]
        # add regression score if it is available
        if 'regression_score' in dramas_df.columns:
            filtered_dramas_df = dramas_df[(dramas_df['regression_score'] > min_regression_score)]

        # count number of generations
        number_of_generations = 0
        for gen in filtered_dramas_df['filename']:
            number_of_generations += 1

        # write to file
        with open(output_file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            # only write in trial 1
            if trial == 1 and 'regression_score' in dramas_df.columns:
                writer.writerow(['generated_filenames', 'Minimal_Mean_Sentence_Length', 'Occurrences',
                                 'Min_Number_of_Speakers', 'Max_Number_of_Speakers', 'Minimal_annotation_1',
                                 'Maximal_annotation_1', 'Minimal_annotation_2', 'Maximal_annotation_2',
                                 'min_speech_length_span', 'max_speech_length_span', 'min_speech_length_stdev',
                                 'max_speech_length_stdev', 'min_food_verbs', 'food_noun_count', 'min_regression_score', 'trial',
                                 'num_generations'])

                for gen in filtered_dramas_df['filename']:
                    writer.writerow([gen, minimal_mean_sentence_length, occurrences, min_num_speakers, max_num_speakers,
                                     min_humor_ann_1, max_humor_ann_1, min_humor_ann_2, max_humor_ann_2,
                                     min_speech_length_span, max_speech_length_span, min_speech_length_stdev,
                                     max_speech_length_stdev, min_food_verbs, min_food_nouns, min_regression_score, trial,
                                     number_of_generations])
            elif trial == 1:
                writer.writerow(['generated_filenames', 'Minimal_Mean_Sentence_Length', 'Occurrences',
                                 'Min_Number_of_Speakers', 'Max_Number_of_Speakers', 'Minimal_annotation_1',
                                 'Maximal_annotation_1', 'Minimal_annotation_2', 'Maximal_annotation_2',
                                 'min_speech_length_span', 'max_speech_length_span', 'min_speech_length_stdev',
                                 'max_speech_length_stdev', 'min_food_verbs', 'food_noun_count', 'trial',
                                 'num_generations'])

                for gen in filtered_dramas_df['filename']:
                    writer.writerow([gen, minimal_mean_sentence_length, occurrences, min_num_speakers, max_num_speakers,
                                     min_humor_ann_1, max_humor_ann_1, min_humor_ann_2, max_humor_ann_2,
                                     min_speech_length_span, max_speech_length_span, min_speech_length_stdev,
                                     max_speech_length_stdev, min_food_verbs, min_food_nouns, trial,
                                     number_of_generations])

            elif 'regression_score' in dramas_df.columns:
                for gen in filtered_dramas_df['filename']:
                    writer.writerow([gen, minimal_mean_sentence_length, occurrences, min_num_speakers, max_num_speakers,
                                     min_humor_ann_1, max_humor_ann_1, min_humor_ann_2, max_humor_ann_2,
                                     min_speech_length_span, max_speech_length_span, min_speech_length_stdev,
                                     max_speech_length_stdev, min_food_verbs, min_food_nouns, min_regression_score,
                                     trial, number_of_generations])
            else:
                for gen in filtered_dramas_df['filename']:
                    writer.writerow([gen, minimal_mean_sentence_length, occurrences, min_num_speakers, max_num_speakers,
                                     min_humor_ann_1, max_humor_ann_1, min_humor_ann_2, max_humor_ann_2,
                                     min_speech_length_span, max_speech_length_span, min_speech_length_stdev,
                                     max_speech_length_stdev, min_food_verbs, min_food_nouns, trial,
                                     number_of_generations])

    # ------------------------------------------------------------------------------------------------------------------
    # else if we do not have annotations
    else:
        filtered_dramas_df = dramas_df[(dramas_df['mean_sentence_length'] > minimal_mean_sentence_length) &
                                       (dramas_df['double_occurrence'] == occurrences) &
                                       (dramas_df['num_persons'] > min_num_speakers) &
                                       (dramas_df['num_persons'] < max_num_speakers) &
                                       (dramas_df['speech_length_span'] > min_speech_length_span) &
                                       (dramas_df['speech_length_span'] < max_speech_length_span) &
                                       (dramas_df['speech_length_stdev'] > min_speech_length_stdev) &
                                       (dramas_df['speech_length_stdev'] < max_speech_length_stdev) &
                                       (dramas_df['food_verb_count'] > min_food_verbs) &
                                       (dramas_df['food_noun_count'] > min_food_verbs)
                                       ]
        # add regression score if it is available
        if 'regression_score' in dramas_df.columns:
            filtered_dramas_df = dramas_df[(dramas_df['regression_score'] > min_regression_score)]

        # count number of generations
        number_of_generations = 0
        for gen in filtered_dramas_df['filename']:
            number_of_generations += 1

        # write to file
        with open(output_file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            # only write in trial 1
            if trial == 1 and 'regression_score' in dramas_df.columns:
                writer.writerow(['generated_filenames', 'Minimal_Mean_Sentence_Length', 'Occurrences',
                                 'Min_Number_of_Speakers', 'Max_Number_of_Speakers',
                                 'min_speech_length_span', 'max_speech_length_span', 'min_speech_length_stdev',
                                 'max_speech_length_stdev', 'min_food_verbs', 'food_noun_count',  'min_regression_score',
                                 'trial', 'num_generations'])

                for gen in filtered_dramas_df['filename']:
                    writer.writerow([gen, minimal_mean_sentence_length, occurrences, min_num_speakers, max_num_speakers,
                                     min_speech_length_span, max_speech_length_span, min_speech_length_stdev,
                                     max_speech_length_stdev, min_food_verbs, min_food_nouns, min_regression_score, trial,
                                     number_of_generations])
            elif trial == 1:
                writer.writerow(['generated_filenames', 'Minimal_Mean_Sentence_Length', 'Occurrences',
                                 'Min_Number_of_Speakers', 'Max_Number_of_Speakers',
                                 'min_speech_length_span', 'max_speech_length_span', 'min_speech_length_stdev',
                                 'max_speech_length_stdev', 'min_food_verbs', 'food_noun_count', 'trial',
                                 'num_generations'])

                for gen in filtered_dramas_df['filename']:
                    writer.writerow([gen, minimal_mean_sentence_length, occurrences, min_num_speakers, max_num_speakers,
                                     min_speech_length_span, max_speech_length_span, min_speech_length_stdev,
                                     max_speech_length_stdev, min_food_verbs, min_food_nouns, trial,
                                     number_of_generations])

            elif 'regression_score' in dramas_df.columns:
                for gen in filtered_dramas_df['filename']:
                    writer.writerow([gen, minimal_mean_sentence_length, occurrences, min_num_speakers, max_num_speakers,
                                     min_speech_length_span, max_speech_length_span, min_speech_length_stdev,
                                     max_speech_length_stdev, min_food_verbs, min_food_nouns, min_regression_score,
                                     trial, number_of_generations])

            else:
                for gen in filtered_dramas_df['filename']:
                    writer.writerow([gen, minimal_mean_sentence_length, occurrences, min_num_speakers, max_num_speakers,
                                     min_speech_length_span, max_speech_length_span, min_speech_length_stdev,
                                     max_speech_length_stdev, min_food_verbs, min_food_nouns, trial,
                                     number_of_generations])

    if number_of_generations > 0:
        print("trial and number of generations: ", str(trial), str(number_of_generations))


def humor_annotations(dramas_df, cwd):
    '''
    humor_annotations
    Reads in the humor annotations from a csv file (hardcoded because we only have one)
    :param: dramas_df: pandas dataframe containing all dramas with their data
            cwd: for file I/O
    :return: the dataframe with one added column for the annotations
    Author: Charlotte
    '''
    cwd = os.path.dirname(cwd) + '/Annotationen Humor_Rieger.csv'
    print(cwd)
    humor = pd.read_csv(cwd, delimiter=';')
    selected_columns = ["Generation", "Annotation_1", "Annotation_2"]
    humor = humor[selected_columns]
    merged_df = dramas_df.merge(humor[['Generation', 'Annotation_1', 'Annotation_2']], left_on='filename', right_on='Generation',
                                how='left')
    merged_df = merged_df.drop('Generation', axis=1)
    return merged_df


def loop_evaluation(dramas_df, path, annotations = False):
    '''
    loop_evaluation
    Loops the best_evaluation method so it does not have to be done manually. Takes all parameter combinations.
    Runtime locally ca. 100 minutes.
    :param: dramas_df: pandas dataframe containing all dramas with their data
            path: path to write results of best_evaluation to.
            annotations: boolean, set true if you have a dataframe with annotations.
    :return: the dataframe with one added column for the annotations
    Author: Charlotte
    '''
    word_counts = dramas_df['raw_text'].apply(lambda x: len(x.split()))
    mean_word_count = word_counts.mean()

    if annotations:
        minimal_sentence_length = [1, 10, 15, 20]
        occurrences = [0, 1]
        min_num_speakers = [0, 1, 2]
        max_num_speakers = [5, 4, 3]
        min_humor_ann_1 = [1, 2]
        max_humor_ann_1 = [3]
        min_humor_ann_2 = [1, 2]
        max_humor_ann_2 = [3]
        min_speech_length_span = [0, 10]
        max_speech_length_span = [100]
        min_speech_length_stdev = [0, 20, 50]
        max_speech_length_stdev = [60, 80, 100]
        min_food_verbs = [round(mean_word_count*1/20), round(mean_word_count*1/30), round(mean_word_count*1/40)]
        min_food_nouns = [round(mean_word_count*1/10), round(mean_word_count*1/20), round(mean_word_count*1/30)]
        min_regression_score = [0.0, 0.5, 0.8, 1.0, 1.5]
        # Generate all combinations of parameter values
        parameter_combinations = itertools.product(
            minimal_sentence_length,
            occurrences,
            min_num_speakers,
            max_num_speakers,
            min_humor_ann_1,
            max_humor_ann_1,
            min_humor_ann_2,
            max_humor_ann_2,
            min_speech_length_span,
            max_speech_length_span,
            min_speech_length_stdev,
            max_speech_length_stdev,
            min_food_verbs,
            min_food_nouns,
            min_regression_score
        )

        trial = 1
        # Loop through the combinations and call best_generations function
        for combination in parameter_combinations:
            best_generations(dramas_df, path, *combination, trial=trial)
            trial += 1
    else:
        minimal_sentence_length = [1, 10, 15, 20]
        occurrences = [0, 1]
        min_num_speakers = [0, 1, 2]
        max_num_speakers = [5, 4, 3]
        min_speech_length_span = [0, 10]
        max_speech_length_span = [100]
        min_speech_length_stdev = [0, 20, 50]
        max_speech_length_stdev = [60, 80, 100]
        min_food_verbs = [round(mean_word_count * 1 / 20), round(mean_word_count * 1 / 30),
                          round(mean_word_count * 1 / 40)]
        min_food_nouns = [round(mean_word_count * 1 / 10), round(mean_word_count * 1 / 20),
                          round(mean_word_count * 1 / 30)]
        min_regression_score = [0.0, 0.5, 0.8, 1.0, 1.5]

        # Generate all combinations of parameter values
        parameter_combinations = itertools.product(
            minimal_sentence_length,
            occurrences,
            min_num_speakers,
            max_num_speakers,
            min_speech_length_span,
            max_speech_length_span,
            min_speech_length_stdev,
            max_speech_length_stdev,
            min_food_verbs,
            min_food_nouns,
            min_regression_score
        )

        trial = 1
        # Loop through the combinations and call best_generations function
        for combination in parameter_combinations:
            best_generations(dramas_df, path, *combination, trial=trial)
            trial += 1


def main():

    pd.set_option('display.max_rows', None)
    path2dir = os.path.dirname(__file__)
    drama_list = string_to_drama_format(path2dir)
    dramas = all_dramas_pd(drama_list)
    dramas = mean_sentence_length(dramas)
    dramas = similar_sentence(dramas)
    dramas = phrasenwiederholung_lexicalrichness(dramas)
    dramas = speech_length_variation(dramas)
    dramas = occurrences(dramas)
    dramas = humor_annotations(dramas, path2dir)
    dramas = speech_length_variation(dramas)
    dramas = food_words(dramas, path2dir)
    dramas = raw_text(dramas)
    cwd = os.path.dirname(path2dir)
    cwd = cwd + '/dramas.csv'
    dramas.to_csv(cwd, index=False, sep='\t')

    word_counts = dramas['raw_text'].apply(lambda x: len(x.split()))
    mean_word_count = word_counts.mean()
    print(mean_word_count) #55.1

    # evaluation part

    # single evaluation
    best_generations(dramas, 'find_optimal_params_03.csv', minimal_mean_sentence_length = 0, occurrences = 1,
                     min_num_speakers = 2,
                    max_num_speakers = 4, min_humor_ann_1 = 0, max_humor_ann_1 = 3, min_humor_ann_2 = 1,
                    max_humor_ann_2 = 3, min_speech_length_span = 0, max_speech_length_span = 100,
                    min_speech_length_stdev = 0, max_speech_length_stdev = 100,
                    min_food_verbs = round(mean_word_count*1/30),
                     min_food_nouns = round(mean_word_count*1/20),
                     min_regression_score = 0.0, trial = 1)

    # automated evaluation, with parameters looped in the function. Annotations parameter needs to be provided
    # loop_evaluation(dramas, 'loop_evaluation_results.csv', annotations= True)



    # plot basic statistics

    plot_mean_persons(dramas)
    plot_mean_sentences(dramas)
    plot_column_of_df(dramas, 'speech_length_span', 'speech_length_span.png')


if __name__ == "__main__":
    main()
