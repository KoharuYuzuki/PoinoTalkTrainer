import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wavfile
import sys
import tempfile
import utils

def main():
  with tempfile.NamedTemporaryFile() as tmp_file:
    config = utils.load_json('config.json')

    text          = sys.argv[1] if len(sys.argv) > 1 else ''
    bin_file_path = config['openjtalk_bin_path']
    dict_dir_path = config['openjtalk_dict_path']
    hts_file_path = config['openjtalk_hts_path']
    lab_file_path = tmp_file.name
    wav_file_path = sys.argv[2] if len(sys.argv) > 2 else ''

    if wav_file_path == '':
      raise ValueError('wav file path is empty')

    duration_model = tf.keras.models.load_model(config['duration_model_save_path_for_py'])
    f0_model       = tf.keras.models.load_model(config['f0_model_save_path_for_py'])
    volume_model   = tf.keras.models.load_model(config['volume_model_save_path_for_py'])

    assert duration_model != None
    assert f0_model != None
    assert volume_model != None

    utils.gen_label(
      text,
      bin_file_path,
      dict_dir_path,
      hts_file_path,
      lab_file_path
    )

    dataset = utils.gen_dataset(
      lab_file_path,
      None,
      config['sliding_window_len'],
      config['wav_fs'],
      config['duration_mag_all'],
      config['duration_mag_phonemes'],
      config['duration_mag_indices'],
      config['f0_envelope_offset'],
      config['f0_envelope_mag'],
      config['f0_envelope_len'],
      config['f0_normalization_max'],
      config['volume_mute_threshold'],
      config['volume_envelope_len'],
      is_mono=False
    )

    if dataset is None:
      raise ValueError('dataset generation failed')

    phoneme_number_segments, accent_segments, _, _, _ = dataset

    duration_segments_predicted = duration_model({
      'in_phoneme_number': phoneme_number_segments
    }, training=False).numpy().astype(np.float16)

    f0_segments_predicted = f0_model({
      'in_phoneme_number': phoneme_number_segments,
      'in_accent': accent_segments
    }, training=False).numpy().astype(np.float16)

    volume_segments_predicted = volume_model({
      'in_phoneme_number': phoneme_number_segments,
      'in_accent': accent_segments
    }, training=False).numpy().astype(np.float16)

    f0_segments_predicted = utils.interp_zeros(
      f0_segments_predicted.reshape(-1)
    ).reshape((len(f0_segments_predicted), -1))

    volume_segments_predicted = utils.interp_zeros(
      volume_segments_predicted.reshape(-1)
    ).reshape((len(volume_segments_predicted), -1))

    duration_segments = duration_segments_predicted.reshape(-1)
    f0_segments       = f0_segments_predicted * config['f0_normalization_max']
    volume_segments   = volume_segments_predicted

    label_parsed, _, _ = utils.parse_label(lab_file_path, is_mono=False)
    phoneme_list = [x['phoneme'] for x in label_parsed]

    wave = utils.synth_voice(
      duration_segments,
      f0_segments,
      volume_segments,
      phoneme_list,
      config['synth_fs'],
      config['speed'],
      config['volume'],
      config['pitch']
    ).astype(np.float32)

    wavfile.write(wav_file_path, config['synth_fs'], wave)

if __name__ == '__main__':
  main()
