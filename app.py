import tensorflow as tf
from tensorflow import keras
import pickle
import pandas as pd
import numpy as np
import librosa
from matplotlib import pyplot as plt
from librosa import display

import streamlit as st

model_dir = 'data/model_data'
test_dir = 'data/valid_test'
image_dir = 'images'

def plot_spectrogram(st, file, sampling_rate=48000, hop_length=512, n_fft=2048):
    signal,_ = librosa.load(test_dir + '/' + file, sr=sampling_rate)
    stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    spectrogram=np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    
    plt.clf()
    librosa.display.specshow(log_spectrogram, sr=sampling_rate, hop_length=hop_length)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    plt.savefig(image_dir + '/' +'mel-spectrogram.png',
                facecolor='white',
                transparent=False,
                bbox_inches="tight")
    
    st.write('Mel Spectrogram:')
    st.image(image_dir + '/' + 'mel-spectrogram.png')

def plot_mfcc(st,file, sampling_rate=48000, n_fft=2048, n_mfcc=20, hop_length=512):
    signal,_ = librosa.load(test_dir + '/' + file, sr=sampling_rate)
    mfcc = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    #print('MFCC SHAPE:{}'.format(mfcc.shape))
    plt.clf()
    librosa.display.specshow(mfcc, sr=sampling_rate, hop_length=hop_length)
    plt.xlabel('Time')
    plt.ylabel('MFCC')
    plt.colorbar()
    plt.savefig(image_dir + '/' +'mfcc.png',
            facecolor='white',
            transparent=False,
            bbox_inches="tight")
    st.write('MFCC:')
    st.image(image_dir + '/' + 'mfcc.png')

if __name__ == "__main__":
    model = keras.models.load_model(model_dir + '/' + 'fcn_model')
    test_df = pd.read_pickle(model_dir + '/' + 'test_df.pikl')
    test_feature_df = pd.read_pickle(model_dir + '/' + 'test_feature_df.pikl')
    scaler = None
    encoder = None
    with open(model_dir + '/' + 'fcn_scaler', 'rb') as f:
        scaler = pickle.load(f)
    with open(model_dir + '/' + 'fcn_label_encoder', 'rb') as f:
        encoder = pickle.load(f)
    test_df = test_df[test_df.gender != 'other']
    test_feature_df = test_feature_df[test_feature_df.file.isin(list(test_df.file))]

    sentences = list(test_df.sentence)
    genders = list(test_df.gender)
    files = list(test_df.file)

    selection_list = [' '] + [genders[k] + ' - ' + sentences[k][:25] + '...' for k in range(len(sentences))]

    selected = st.sidebar.selectbox('Pick your audio', selection_list)

    st.title("Simple Neural Net based Gender Prediction for Voice samples")
    file = None
    if selected != '  ':
        file = files[selection_list.index(selected) - 1]
    if file:
        st.audio(test_dir + '/' + file)
        st.sidebar.write('You picked : {}'.format(selected))

        record = test_df[test_df.file == file]
        gender = record.gender.values[0]
        sentence = record.sentence.values[0]
        age = record.age.values[0]
        accent = record.accent.values[0]
        st.sidebar.write('Gender : {}'.format(gender))
        st.sidebar.write("Sentence: {}".format(sentence))
        st.sidebar.write("Age : {}".format(age))
        st.sidebar.write("Accent: {}".format(accent))
        plt_type = st.sidebar.radio("Pick a plot", ["Spectrogram", "MFCC"])

        X_features = test_feature_df[test_feature_df.file == file].drop('file', axis=1)
        X_scaled = scaler.transform(X_features.values.reshape(1,-1))
        y_sample_pred = model.predict(X_scaled)

        classes_y = np.argmax(y_sample_pred, axis=1)
        y_labels = encoder.inverse_transform(classes_y)

        st.write('Predicted Gender : {}'.format(y_labels))

        if plt_type == 'Spectrogram':
            plot_spectrogram(st, file)
        else:
            plot_mfcc(st, file)




