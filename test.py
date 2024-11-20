# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
# from tensorflow.keras.optimizers import RMSprop

# model = Sequential([
#     Input(shape=(300, 150, 3)),
#     Conv2D(16, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(13, activation='softmax')
# ])

# model.compile(
#     optimizer=RMSprop(learning_rate=0.001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# print("Model compiled successfully!")

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("hi")