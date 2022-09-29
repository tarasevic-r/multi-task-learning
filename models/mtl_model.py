from tensorflow.keras.layers import Dense
import keras
from keras.layers import Input, Dense, Multiply
from keras import Model


def build_model(input_shape=9, output_shape=1):
    
    # Model input
    input_layer = Input(shape=input_shape)
    # shared variables X = [x1, x2, ..., x6]
    X = input_layer[:, :6]
    # z variable for task 1
    z = input_layer[:, 6:7]
    # task masks
    t1_mask = input_layer[:, 7:8]
    t2_mask = input_layer[:, 8:9]

    # shared layer and task 2 output
    Xb1 = Dense(output_shape, name='Xb1')(X)
	
    # concatenate X with z for task 1
    Xb2_b4_b3z = keras.layers.concatenate([Xb1, z], name='Xb2_b4_b3z', axis=-1)

    # task 1 output
    y1_pred = Dense(output_shape, activation='sigmoid')(Xb2_b4_b3z)
    # task 1 masked output
    y1_pred_masked = Multiply(name='task_1')([y1_pred, t1_mask])
    
    # task 2 masked output
    y2_pred_masked = Multiply(name='task_2')([Xb1, t2_mask])

    # Define model input and outputs according to task
    model = Model(inputs=input_layer, outputs=[y1_pred_masked, y2_pred_masked])

    optimizer = keras.optimizers.Adam(learning_rate=0.02)
    # Compile model with defined specific loss and loss weights for each task
    model.compile(optimizer=optimizer,
                  loss={
                      'task_1': 'binary_crossentropy',
                      'task_2': 'mse',
                  },
                  loss_weights={'task_1': 0.9,
                                'task_2': 0.1},
                  metrics=["accuracy"]
                  # run_eagerly=True
                  )
    # model.summary()

    return model
