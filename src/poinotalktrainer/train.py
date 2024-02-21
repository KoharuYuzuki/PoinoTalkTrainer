import tensorflow as tf
import tensorflowjs as tfjs
import utils

def train_duration_model(
  dataset: dict,
  config: dict
) -> None:
  print('\n[duration model training]\n')

  epochs          = config['duration_model_epochs']
  batch_size      = config['duration_model_batch_size']
  save_dir_for_py = config['duration_model_save_dir_for_py']
  save_dir_for_js = config['duration_model_save_dir_for_js']

  layer_in_phoneme_number_shape = dataset['phoneme_number'].shape[1:]
  layer_mid_units               = config['duration_model_mid_layer_units']
  layer_out_duration_units      = dataset['duration'].shape[1]

  layer_in_phoneme_number = tf.keras.layers.Input(layer_in_phoneme_number_shape, name='in_phoneme_number')
  layer_mid_1             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_in_phoneme_number)
  layer_mid_2             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_1)
  layer_mid_3             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_2)
  layer_mid_4             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_3)
  layer_mid_5             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_4)
  layer_mid_6             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_5)
  layer_out_duration      = tf.keras.layers.Dense(layer_out_duration_units, activation='relu', name='out_duration')(layer_mid_6)

  model = tf.keras.Model(
    [layer_in_phoneme_number],
    [layer_out_duration]
  )
  model.compile(optimizer='adam', loss='huber')
  model.fit(
    { 'in_phoneme_number': dataset['phoneme_number'] },
    { 'out_duration': dataset['duration'] },
    epochs=epochs,
    batch_size=batch_size
  )
  utils.mkdir(save_dir_for_py)
  utils.mkdir(save_dir_for_js)
  model.save(save_dir_for_py)
  tfjs.converters.save_keras_model(model, save_dir_for_js)

def train_f0_model(
  dataset: dict,
  config: dict
) -> None:
  print('\n[f0 model training]\n')

  epochs          = config['f0_model_epochs']
  batch_size      = config['f0_model_batch_size']
  save_dir_for_py = config['f0_model_save_dir_for_py']
  save_dir_for_js = config['f0_model_save_dir_for_js']

  layer_in_phoneme_number_shape = dataset['phoneme_number'].shape[1:]
  layer_in_accent_shape         = dataset['accent'].shape[1:]
  layer_mid_units               = config['f0_model_mid_layer_units']
  layer_out_f0_units            = dataset['f0'].shape[1]

  layer_in_phoneme_number = tf.keras.layers.Input(layer_in_phoneme_number_shape, name='in_phoneme_number')
  layer_in_accent         = tf.keras.layers.Input(layer_in_accent_shape, name='in_accent')
  layer_concat            = tf.keras.layers.Concatenate()([layer_in_phoneme_number, layer_in_accent])
  layer_mid_1             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_concat)
  layer_mid_2             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_1)
  layer_mid_3             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_2)
  layer_mid_4             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_3)
  layer_mid_5             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_4)
  layer_mid_6             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_5)
  layer_out_f0            = tf.keras.layers.Dense(layer_out_f0_units, activation='relu', name='out_f0')(layer_mid_6)

  model = tf.keras.Model(
    [layer_in_phoneme_number, layer_in_accent],
    [layer_out_f0]
  )
  model.compile(optimizer='adam', loss='huber')
  model.fit(
    { 'in_phoneme_number': dataset['phoneme_number'], 'in_accent': dataset['accent'] },
    { 'out_f0': dataset['f0'] },
    epochs=epochs,
    batch_size=batch_size
  )
  utils.mkdir(save_dir_for_py)
  utils.mkdir(save_dir_for_js)
  model.save(save_dir_for_py)
  tfjs.converters.save_keras_model(model, save_dir_for_js)

def train_volume_model(
  dataset: dict,
  config: dict
) -> None:
  print('\n[volume model training]\n')

  epochs          = config['volume_model_epochs']
  batch_size      = config['volume_model_batch_size']
  save_dir_for_py = config['volume_model_save_dir_for_py']
  save_dir_for_js = config['volume_model_save_dir_for_js']

  layer_in_phoneme_number_shape = dataset['phoneme_number'].shape[1:]
  layer_in_accent_shape         = dataset['accent'].shape[1:]
  layer_mid_units               = config['volume_model_mid_layer_units']
  layer_out_volume_units        = dataset['volume'].shape[1]

  layer_in_phoneme_number = tf.keras.layers.Input(layer_in_phoneme_number_shape, name='in_phoneme_number')
  layer_in_accent         = tf.keras.layers.Input(layer_in_accent_shape, name='in_accent')
  layer_concat            = tf.keras.layers.Concatenate()([layer_in_phoneme_number, layer_in_accent])
  layer_mid_1             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_concat)
  layer_mid_2             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_1)
  layer_mid_3             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_2)
  layer_mid_4             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_3)
  layer_mid_5             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_4)
  layer_mid_6             = tf.keras.layers.Dense(layer_mid_units, activation='relu')(layer_mid_5)
  layer_out_volume        = tf.keras.layers.Dense(layer_out_volume_units, activation='relu', name='out_volume')(layer_mid_6)

  model = tf.keras.Model(
    [layer_in_phoneme_number, layer_in_accent],
    [layer_out_volume]
  )
  model.compile(optimizer='adam', loss='huber')
  model.fit(
    { 'in_phoneme_number': dataset['phoneme_number'], 'in_accent': dataset['accent'] },
    { 'out_volume': dataset['volume'] },
    epochs=epochs,
    batch_size=batch_size
  )
  utils.mkdir(save_dir_for_py)
  utils.mkdir(save_dir_for_js)
  model.save(save_dir_for_py)
  tfjs.converters.save_keras_model(model, save_dir_for_js)

def main():
  config = utils.load_json('config.json')
  dataset = utils.load_data('lab/*.lab', 'wav/*.wav', config)

  utils.set_seed(config['seed'])

  train_duration_model(dataset, config)
  train_f0_model(dataset, config)
  train_volume_model(dataset, config)

if __name__ == '__main__':
  main()
