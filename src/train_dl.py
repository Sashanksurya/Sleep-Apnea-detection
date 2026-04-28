import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import os

# GPU Guard
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
    except RuntimeError as e: print(e)

def build_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(64, 7, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(4),
        Dropout(0.2),
        Conv1D(128, 5, activation='relu', padding='same'),
        MaxPooling1D(4),
        LSTM(128, dropout=0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

if __name__ == "__main__":
    X_tr, y_tr = np.load("data/processed/X_train.npy"), np.load("data/processed/y_train.npy")
    X_v, y_v = np.load("data/processed/X_val.npy"), np.load("data/processed/y_val.npy")
    
    model = build_cnn_lstm((X_tr.shape[1], X_tr.shape[2]))
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True),
        ModelCheckpoint('model/cnn_lstm_best.keras', save_best_only=True)
    ]
    
    model.fit(X_tr, y_tr, validation_data=(X_v, y_v), epochs=50, batch_size=32, callbacks=callbacks)
    model.save("model/cnn_lstm_final.keras")
    print("✅ Phase 5 Complete: Deep Learning model saved.")