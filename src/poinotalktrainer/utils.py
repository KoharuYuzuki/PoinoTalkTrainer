import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wavfile
import pyworld as pw
import math
import re
import json
import os
import random
from glob import glob
from tqdm import tqdm
from numpy.typing import NDArray

def compute_seq2seg_len(
  sequence_len: int,
  seg_len: int,
  hop_len: int
) -> int:
  if (sequence_len >= seg_len) and (hop_len > 0):
    seguments_len = math.ceil((sequence_len / hop_len) - (seg_len / hop_len) + 1)
  else:
    seguments_len = 1

  return seguments_len

def compute_seg2seq_len(
  seguments_len: int,
  seg_len: int,
  hop_len: int
) -> int:
  sequence_len = hop_len * (seguments_len - 1) + seg_len
  return sequence_len

def seq2seg(
  sequence: NDArray,
  seg_len: int,
  hop_len: int,
  apply_window: bool = False
) -> NDArray:
  sequence_len = len(sequence)
  seguments_len = compute_seq2seg_len(sequence_len, seg_len, hop_len)
  seguments = np.zeros(
    (seguments_len, seg_len, *sequence.shape[1:]),
    dtype=sequence.dtype
  )

  for i in range(seguments_len):
    start = hop_len * i
    end = start + seg_len

    segument = sequence[start:end]
    segument_len = len(segument)

    if segument_len < seg_len:
      segument = np.concatenate((
        segument,
        np.zeros(
          ((seg_len - segument_len), *sequence.shape[1:]),
          dtype=sequence.dtype
        )
      ))

    seguments[i] = segument

  if apply_window:
    seguments *= np.hanning(seguments.shape[-1])

  return seguments

def seg2seq(
  seguments: NDArray,
  seg_len: int,
  hop_len: int,
  adj_overlap_value: bool = False
) -> NDArray:
  seguments_len = len(seguments)
  sequence_len = compute_seg2seq_len(seguments_len, seg_len, hop_len)
  sequence = np.zeros(
    (sequence_len, *seguments.shape[2:]),
    dtype=seguments.dtype
  )
  adjuster = (hop_len / seg_len) * 2

  for i in range(seguments_len):
    start = hop_len * i
    end = start + seg_len
    segument = seguments[i]

    if adj_overlap_value:
      segument *= adjuster

    sequence[start:end] += segument

  return sequence

def resample(
  y: NDArray,
  num: int
) -> NDArray:
  assert len(y.shape) == 1
  x = np.linspace(0, 1, num=len(y))
  new_x = np.linspace(0, 1, num=num)
  new_y = np.interp(new_x, x, y)
  return new_y

def detect_f0(
  wave: NDArray,
  fs: float,
  frame_period: int = 5
) -> NDArray:
  wave_f64 = wave.astype(np.float64)
  f0, t = pw.dio(wave_f64, fs, frame_period=frame_period)
  f0 = pw.stonemask(wave_f64, f0, t, fs)
  f0 = f0.astype(wave.dtype)
  return f0

def detect_volume(
  wave: NDArray,
  fs: float
) -> NDArray:
  seg_len = int(fs * 0.01)
  hop_len = seg_len // 4

  seguments = seq2seg(wave, seg_len, hop_len, apply_window=True)
  for segument in seguments:
    volume = np.max(np.abs(segument))
    segument.fill(volume)
  seguments *= np.hanning(seg_len)

  sequence = seg2seq(seguments, seg_len, hop_len, adj_overlap_value=True)
  return sequence[:len(wave)]

def interp_zeros(
  data: NDArray
) -> NDArray:
  data_len = len(data)
  data_copied = np.copy(data)

  nonzero_index = np.nonzero(data_copied)[0]
  nonzero_index_len = len(nonzero_index)

  for i, index in enumerate(nonzero_index):
    if i == 0:
      if index != 0:
        data_copied[:index] = np.linspace(data_copied[index], data_copied[index], num=index)
      continue

    if (i >= (nonzero_index_len - 1)) and (nonzero_index_len < data_len):
      data_copied[index:] = np.linspace(data_copied[index], data_copied[index], num=data_len - index)

    prev_index = nonzero_index[i - 1]
    if prev_index == (index - 1):
      continue

    prev_value = data_copied[prev_index]
    value = data_copied[index]

    data_copied[prev_index + 1:index] = np.linspace(prev_value, value, num=index - (prev_index + 1))

  return data_copied

def parse_label(
  label_file_path: str,
  is_mono: bool
) -> tuple | None:
  with open(label_file_path, mode='r', encoding='utf-8') as file:
    label = file.read()

  if is_mono:
    lab_pattern = re.compile('^(?P<start>[0-9]+) (?P<end>[0-9]+) (?P<phoneme>[a-z]+)', re.I | re.M)
  else:
    lab_pattern = re.compile('^(?P<start>[0-9]+) (?P<end>[0-9]+) [a-z]+\\^[a-z]+-(?P<phoneme>[a-z]+)\\+[a-z]+=[a-z]+\\/A:(?P<accent_pos>-*[0-9|a-z]+)\\+(?P<accent_num_1>[0-9|a-z]+)\\+(?P<accent_num_2>[0-9|a-z]+)', re.M | re.I)

  results = [x.groupdict() for x in re.finditer(lab_pattern, label)]
  min_results_len = 3

  if len(results) < min_results_len:
    return None

  to_sec = 1e-7
  label_parsed = []

  for result in results:
    start = int(result['start']) * to_sec
    end = int(result['end']) * to_sec
    phoneme = result['phoneme'].lower() if result['phoneme'] != 'N' else result['phoneme']

    label_parsed.append({
      'start': start,
      'end': end,
      'phoneme': phoneme,
      'accent': None
    })

  sil_end_sec = label_parsed[0]['end']
  sil_start_sec = label_parsed[-1]['start']

  results = results[1:-1]
  label_parsed = label_parsed[1:-1]

  if not is_mono:
    prev_accent_num_1 = 2
    prev_accent_num_2 = 1
    accent_groups = []
    label_index = 0

    for result in results:
      phoneme = result['phoneme']
      accent_pos = int(result['accent_pos']) if result['accent_pos'] != 'xx' else None
      accent_num_1 = int(result['accent_num_1']) if result['accent_num_1'] != 'xx' else None
      accent_num_2 = int(result['accent_num_2']) if result['accent_num_2'] != 'xx' else None

      if (
        (accent_num_1 == None) or
        (accent_num_2 == None) or
        (prev_accent_num_1 > accent_num_1) or
        (prev_accent_num_2 < accent_num_2)
      ):
        accent_groups.append([])

      accent_groups[-1].append({
        'phoneme': phoneme,
        'accent_pos': accent_pos
      })

      prev_accent_num_1 = accent_num_1 if accent_num_1 != None else 2
      prev_accent_num_2 = accent_num_2 if accent_num_2 != None else 1

    for group in accent_groups:
      accent_pos_list = [x['accent_pos'] for x in group]
      accent_set = sorted(list(set(accent_pos_list)))

      if None in accent_set:
        accents = [0 for _ in range(len(accent_set))]
      else:
        index = accent_set.index(0)

        if index == 0:
          accents = [1] + [0 for _ in range(len(accent_set) - 1)]
        elif index >= 1:
          accents = [0] + [1 for _ in range(index)] + [0 for _ in range(len(accent_set) - (index + 1))]
        else:
          accents = [0 for _ in range(len(accent_set))]

      prev_accent_pos = None

      for accent_pos in accent_pos_list:
        if (prev_accent_pos == accent_pos) and (accent_pos != None):
          continue

        accent_num = accent_pos_list.count(accent_pos)
        accent = accents.pop(0)

        for _ in range(accent_num):
          label_parsed[label_index]['accent'] = accent
          label_index += 1

        prev_accent_pos = accent_pos

  return (
    label_parsed,
    sil_end_sec,
    sil_start_sec
  )

def gen_dataset(
  lab_file_path: str,
  wav_file_path: str | None,
  sliding_window_len: int,
  fs: float,
  duration_mag_all: float,
  duration_mag_phonemes: dict,
  f0_envelope_offset: float,
  f0_envelope_mag: float,
  f0_envelope_len: int,
  f0_normalization_max: float,
  volume_envelope_len: int,
  is_mono: bool
) -> tuple[
  NDArray,
  NDArray | None,
  NDArray,
  NDArray | None,
  NDArray | None
] | None:
  phoneme_list = [
    'sil',
    'pau',

    'a',
    'i',
    'u',
    'e',
    'o',

    'N',

    'k',
    'kw',
    'ky',

    's',
    'sh',

    't',
    'ts',
    'ty',

    'ch',

    'n',
    'ny',

    'h',
    'hy',

    'm',
    'my',

    'y',

    'r',
    'ry',

    'w',

    'b',
    'by',

    'd',
    'dy',

    'g',
    'gw',
    'gy',

    'j',

    'v',

    'f',

    'z',

    'p',
    'py',

    'cl'
  ]

  parsed = parse_label(lab_file_path, is_mono)
  if parsed == None:
    return None

  label_parsed, sil_end_sec, sil_start_sec = parsed

  invalid_value  = -1
  num_before     = math.ceil((sliding_window_len - 1) / 2)
  num_after      = math.floor((sliding_window_len - 1) / 2)

  padding_before = [invalid_value for _ in range(num_before)]
  padding_after  = [invalid_value for _ in range(num_after)]

  phoneme_number = np.array(
    [phoneme_list.index(x['phoneme']) / len(phoneme_list) for x in label_parsed],
    dtype=np.float16
  )

  durations = np.array(
    [x['end'] - x['start'] for x in label_parsed],
    dtype=np.float16
  )

  if wav_file_path == None:
    f0_seguments = None
    volume_seguments = None

    if is_mono:
      accent_seguments = None
    else:
      accent_array = np.array(padding_before + [x['accent'] for x in label_parsed] + padding_after, dtype=np.float16)
      accent_seguments = seq2seg(accent_array, sliding_window_len, 1)
  else:
    wave_fs, wave = wavfile.read(wav_file_path)
    assert wave_fs == fs

    if (
      (wave.dtype != np.float16) and
      (wave.dtype != np.float32) and
      (wave.dtype != np.float64) and
      (wave.dtype != np.float128)
    ):
      wave = wave.astype(np.float16)
      wave = wave / np.iinfo(wave.dtype).max
    else:
      wave = wave.astype(np.float16)

    wave = wave[int(sil_end_sec * fs):int(sil_start_sec * fs)]

    f0_envelope = detect_f0(wave, fs, frame_period=50)
    f0_envelope = interp_zeros(f0_envelope)

    f0_envelope[f0_envelope != 0] *= f0_envelope_mag
    f0_envelope[f0_envelope != 0] += f0_envelope_offset
    f0_envelope[f0_envelope != 0] /= f0_normalization_max

    f0_envelope_min = np.min(f0_envelope[f0_envelope != 0])
    f0_envelope_max = np.max(f0_envelope[f0_envelope != 0])
    f0_envelope_center = f0_envelope_min + ((f0_envelope_max - f0_envelope_min) / 2)

    f0_envelope = resample(f0_envelope, len(wave))

    volume_envelope = detect_volume(wave, fs)

    prev_sec = 0
    accents = []
    f0_seguments = np.zeros((len(durations), f0_envelope_len), dtype=np.float16)
    volume_seguments = np.zeros((len(durations), volume_envelope_len), dtype=np.float16)

    for i, duration in enumerate(durations):
      start_sec = prev_sec
      end_sec = start_sec + duration

      start_index = int(start_sec * fs)
      end_index = int(end_sec * fs)

      f0_segument = f0_envelope[start_index:end_index]
      f0_segument = resample(f0_segument, f0_envelope_len)
      f0_seguments[i] = f0_segument

      accent = 1 if np.average(f0_segument[f0_segument != 0]) >= f0_envelope_center else 0
      accents.append(accent)

      volume_segument = volume_envelope[start_index:end_index]
      volume_segument = resample(volume_segument, volume_envelope_len)
      volume_seguments[i] = volume_segument

      prev_sec = end_sec

    accent_array = np.array(padding_before + accents + padding_after, dtype=np.float16)
    accent_seguments = seq2seg(accent_array, sliding_window_len, 1)

  for key in duration_mag_phonemes.keys():
    if not key in phoneme_list:
      continue

    index = phoneme_list.index(key)
    number = np.float16(index / len(phoneme_list))
    mag = duration_mag_phonemes[key]

    bool_array = phoneme_number == number
    durations[bool_array] *= mag

  durations *= duration_mag_all

  phoneme_number_seguments = seq2seg(
    np.concatenate((padding_before, phoneme_number, padding_after)),
    sliding_window_len,
    1
  )

  duration_seguments = durations.reshape((-1, 1))

  return (
    phoneme_number_seguments,
    accent_seguments,
    duration_seguments,
    f0_seguments,
    volume_seguments
  )

def load_data(
  lab_file_glob_pattern: str,
  wav_file_glob_pattern: str,
  config: dict
) -> dict:
  lab_file_paths = sort_glob(lab_file_glob_pattern)
  wav_file_paths = sort_glob(wav_file_glob_pattern)

  assert len(lab_file_paths) == len(wav_file_paths)
  files_len = len(lab_file_paths)

  phoneme_number_seguments_list = []
  accent_seguments_list         = []
  duration_seguments_list       = []
  f0_seguments_list             = []
  volume_seguments_list         = []

  for i in tqdm(range(files_len), desc='[data loading]'):
    lab_file_path = lab_file_paths[i]
    wav_file_path = wav_file_paths[i]

    dataset = gen_dataset(
      lab_file_path,
      wav_file_path,
      config['sliding_window_len'],
      config['wav_fs'],
      config['duration_mag_all'],
      config['duration_mag_phonemes'],
      config['f0_envelope_offset'],
      config['f0_envelope_mag'],
      config['f0_envelope_len'],
      config['f0_normalization_max'],
      config['volume_envelope_len'],
      True
    )

    if (dataset is None) or any(x is None for x in dataset):
      continue

    (
      phoneme_number_seguments,
      accent_seguments,
      duration_seguments,
      f0_seguments,
      volume_seguments
    ) = dataset

    phoneme_number_seguments_list += list(phoneme_number_seguments)
    accent_seguments_list         += list(accent_seguments)
    duration_seguments_list       += list(duration_seguments)
    f0_seguments_list             += list(f0_seguments)
    volume_seguments_list         += list(volume_seguments)

  phoneme_number_seguments = np.array(phoneme_number_seguments_list, dtype=np.float16)
  accent_seguments         = np.array(accent_seguments_list, dtype=np.float16)
  duration_seguments       = np.array(duration_seguments_list, dtype=np.float16)
  f0_seguments             = np.array(f0_seguments_list, dtype=np.float16)
  volume_seguments         = np.array(volume_seguments_list, dtype=np.float16)

  return {
    'phoneme_number': phoneme_number_seguments,
    'accent':         accent_seguments,
    'duration':       duration_seguments,
    'f0':             f0_seguments,
    'volume':         volume_seguments
  }

def load_json(
  json_file_path: str
) -> any:
  with open(json_file_path, mode='r', encoding='utf-8') as file:
    json_str = file.read()

  return json.loads(json_str)

def sort_glob(
  pattern: str,
) -> list:
  return sorted(glob(pattern))

def set_seed(
  seed: int
) -> None:
  os.environ['PYTHONHASHSEED'] = str(seed)
  tf.random.set_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

def mkdir(
  path: str
) -> None:
  os.makedirs(path, exist_ok=True)
