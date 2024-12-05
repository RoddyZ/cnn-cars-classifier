# 1. Crear una función de modelo
import tensorflow as tf

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units', min_value=32, max_value=256, step=32),  # Número de neuronas
        activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])  # Función de activación
    ))
    model.add(tf.keras.layers.Dense(1, activation='linear'))  # Salida
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd']),  # Optimizador
        loss='mse',
        metrics=['mae']
    )
    return model


# 2. Configurar el tuner
import keras_tuner as kt
tuner = kt.Hyperband(
    build_model,  # Función del modelo
    objective='val_mae',  # Métrica a optimizar
    max_epochs=50,  # Máximo de épocas por prueba
    factor=3,  # Reducción de recursos en Hyperband
    directory='my_dir',  # Carpeta para guardar resultados
    project_name='my_project'  # Nombre del proyecto
)

# 3. Realizar la búsqueda
tuner.search(
    X_train, y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
)
# 4. Obtener los mejores hiperparámetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Mejores hiperparámetros:
- Neuronas: {best_hps.get('units')}
- Activación: {best_hps.get('activation')}
- Optimizador: {best_hps.get('optimizer')}
""")

# Entrenar el modelo con los mejores hiperparámetros
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, validation_split=0.2, epochs=50)

#5. Evaluar el modelo
loss, mae = best_model.evaluate(X_test, y_test)
print(f"MAE en el conjunto de prueba: {mae:.4f}")
