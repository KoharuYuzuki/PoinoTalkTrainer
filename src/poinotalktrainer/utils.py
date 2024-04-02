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

openjtalk_phoneme_list = [
  'sil',
  'pau',
  None,
  'a',
  'i',
  'u',
  'e',
  'o',
  None,
  'N',
  None,
  'k',
  'kw',
  'ky',
  None,
  's',
  'sh',
  None,
  't',
  'ts',
  'ty',
  None,
  'ch',
  None,
  'n',
  'ny',
  None,
  'h',
  'hy',
  None,
  'm',
  'my',
  None,
  'y',
  None,
  'r',
  'ry',
  None,
  'w',
  None,
  'b',
  'by',
  None,
  'd',
  'dy',
  None,
  'g',
  'gw',
  'gy',
  None,
  'j',
  None,
  'v',
  None,
  'f',
  None,
  'z',
  None,
  'p',
  'py',
  None,
  'cl'
]

voice_phoneme_list = [
  'a',
  'i',
  'u',
  'e',
  'o',
  'k',
  's',
  't',
  'n',
  'h',
  'm',
  'y',
  'r',
  'w',
  'g',
  'z',
  'd',
  'b',
  'p',
  'v',
  'q'
]

romaji_kana_list = [
  ['pau', '、'],
  ['kya', 'きゃ'],
  ['kyu', 'きゅ'],
  ['kye', 'きぇ'],
  ['kyo', 'きょ'],
  ['gya', 'ぎゃ'],
  ['gyu', 'ぎゅ'],
  ['gye', 'ぎぇ'],
  ['gyo', 'ぎょ'],
  ['kwa', 'くゎ'],
  ['gwa', 'ぐゎ'],
  ['sha', 'しゃ'],
  ['shi', 'し'],
  ['shu', 'しゅ'],
  ['she', 'しぇ'],
  ['sho', 'しょ'],
  ['cha', 'ちゃ'],
  ['chi', 'ち'],
  ['chu', 'ちゅ'],
  ['che', 'ちぇ'],
  ['cho', 'ちょ'],
  ['tsa', 'つぁ'],
  ['tsi', 'つぃ'],
  ['tsu', 'つ'],
  ['tse', 'つぇ'],
  ['tso', 'つぉ'],
  ['tya', 'てゃ'],
  ['tyu', 'てゅ'],
  ['tyo', 'てょ'],
  ['dya', 'でゃ'],
  ['dyu', 'でゅ'],
  ['dyo', 'でょ'],
  ['nya', 'にゃ'],
  ['nyu', 'にゅ'],
  ['nye', 'にぇ'],
  ['nyo', 'にょ'],
  ['hya', 'ひゃ'],
  ['hyu', 'ひゅ'],
  ['hye', 'ひぇ'],
  ['hyo', 'ひょ'],
  ['bya', 'びゃ'],
  ['byu', 'びゅ'],
  ['bye', 'びぇ'],
  ['byo', 'びょ'],
  ['pya', 'ぴゃ'],
  ['pyu', 'ぴゅ'],
  ['pye', 'ぴぇ'],
  ['pyo', 'ぴょ'],
  ['mya', 'みゃ'],
  ['myu', 'みゅ'],
  ['mye', 'みぇ'],
  ['myo', 'みょ'],
  ['rya', 'りゃ'],
  ['ryu', 'りゅ'],
  ['rye', 'りぇ'],
  ['ryo', 'りょ'],
  ['cl', 'っ'],
  ['ye', 'いぇ'],
  ['ka', 'か'],
  ['ki', 'き'],
  ['ku', 'く'],
  ['ke', 'け'],
  ['ko', 'こ'],
  ['sa', 'さ'],
  ['si', 'すぃ'],
  ['su', 'す'],
  ['se', 'せ'],
  ['so', 'そ'],
  ['ta', 'た'],
  ['ti', 'てぃ'],
  ['tu', 'とぅ'],
  ['te', 'て'],
  ['to', 'と'],
  ['na', 'な'],
  ['ni', 'に'],
  ['nu', 'ぬ'],
  ['ne', 'ね'],
  ['no', 'の'],
  ['ha', 'は'],
  ['hi', 'ひ'],
  ['he', 'へ'],
  ['ho', 'ほ'],
  ['ma', 'ま'],
  ['mi', 'み'],
  ['mu', 'む'],
  ['me', 'め'],
  ['mo', 'も'],
  ['ya', 'や'],
  ['yu', 'ゆ'],
  ['yo', 'よ'],
  ['ra', 'ら'],
  ['ri', 'り'],
  ['ru', 'る'],
  ['re', 'れ'],
  ['ro', 'ろ'],
  ['wa', 'わ'],
  ['wi', 'うぃ'],
  ['we', 'うぇ'],
  ['wo', 'うぉ'],
  ['fa', 'ふぁ'],
  ['fi', 'ふぃ'],
  ['fu', 'ふ'],
  ['fe', 'ふぇ'],
  ['fo', 'ふぉ'],
  ['va', 'ゔぁ'],
  ['vi', 'ゔぃ'],
  ['vu', 'ゔ'],
  ['ve', 'ゔぇ'],
  ['vo', 'ゔぉ'],
  ['ga', 'が'],
  ['gi', 'ぎ'],
  ['gu', 'ぐ'],
  ['ge', 'げ'],
  ['go', 'ご'],
  ['za', 'ざ'],
  ['zi', 'ずぃ'],
  ['zu', 'ず'],
  ['ze', 'ぜ'],
  ['zo', 'ぞ'],
  ['ja', 'じゃ'],
  ['ji', 'じ'],
  ['ju', 'じゅ'],
  ['je', 'じぇ'],
  ['jo', 'じょ'],
  ['da', 'だ'],
  ['di', 'でぃ'],
  ['du', 'どぅ'],
  ['de', 'で'],
  ['do', 'ど'],
  ['ba', 'ば'],
  ['bi', 'び'],
  ['bu', 'ぶ'],
  ['be', 'べ'],
  ['bo', 'ぼ'],
  ['pa', 'ぱ'],
  ['pi', 'ぴ'],
  ['pu', 'ぷ'],
  ['pe', 'ぺ'],
  ['po', 'ぽ'],
  ['a', 'あ'],
  ['i', 'い'],
  ['u', 'う'],
  ['e', 'え'],
  ['o', 'お'],
  ['N', 'ん']
]

def compute_seq2seg_len(
  sequence_len: int,
  seg_len: int,
  hop_len: int
) -> int:
  if (sequence_len >= seg_len) and (hop_len > 0):
    segments_len = math.ceil((sequence_len / hop_len) - (seg_len / hop_len) + 1)
  else:
    segments_len = 1

  return segments_len

def compute_seg2seq_len(
  segments_len: int,
  seg_len: int,
  hop_len: int
) -> int:
  sequence_len = hop_len * (segments_len - 1) + seg_len
  return sequence_len

def seq2seg(
  sequence: NDArray,
  seg_len: int,
  hop_len: int,
  apply_window: bool = False
) -> NDArray:
  sequence_len = len(sequence)
  segments_len = compute_seq2seg_len(sequence_len, seg_len, hop_len)
  segments = np.zeros(
    (segments_len, seg_len, *sequence.shape[1:]),
    dtype=sequence.dtype
  )

  for i in range(segments_len):
    begin = hop_len * i
    end = begin + seg_len

    segment = sequence[begin:end]
    segment_len = len(segment)

    if segment_len < seg_len:
      segment = np.concatenate((
        segment,
        np.zeros(
          ((seg_len - segment_len), *sequence.shape[1:]),
          dtype=sequence.dtype
        )
      ))

    segments[i] = segment

  if apply_window:
    segments *= np.hanning(segments.shape[-1])

  return segments

def seg2seq(
  segments: NDArray,
  seg_len: int,
  hop_len: int,
  adj_overlap_value: bool = False
) -> NDArray:
  segments_len = len(segments)
  sequence_len = compute_seg2seq_len(segments_len, seg_len, hop_len)
  sequence = np.zeros(
    (sequence_len, *segments.shape[2:]),
    dtype=segments.dtype
  )
  adjuster = (hop_len / seg_len) * 2

  for i in range(segments_len):
    begin = hop_len * i
    end = begin + seg_len
    segment = segments[i]

    if adj_overlap_value:
      segment *= adjuster

    sequence[begin:end] += segment

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
  x = wave.astype(np.float64)
  _f0, t = pw.dio(x, fs, frame_period=frame_period)
  f0 = pw.stonemask(x, _f0, t, fs)
  return f0.astype(wave.dtype)

def detect_volume(
  wave: NDArray,
  fs: float
) -> NDArray:
  seg_len = int(fs * 0.04)
  hop_len = seg_len // 2

  segments = seq2seg(wave, seg_len, hop_len, apply_window=True)
  for segment in segments:
    volume = np.max(np.abs(segment))
    segment.fill(volume)
  segments *= np.hanning(seg_len)

  sequence = seg2seq(segments, seg_len, hop_len, adj_overlap_value=True)
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
  is_mono: bool,
  include_sil: bool = False
) -> tuple | None:
  with open(label_file_path, mode='r', encoding='utf-8') as file:
    label = file.read()

  if is_mono:
    lab_pattern = re.compile('^(?P<begin>[0-9]+) (?P<end>[0-9]+) (?P<phoneme>[a-z]+)', re.I | re.M)
  else:
    lab_pattern = re.compile('^(?P<begin>[0-9]+) (?P<end>[0-9]+) [a-z]+\\^[a-z]+-(?P<phoneme>[a-z]+)\\+[a-z]+=[a-z]+\\/A:(?P<accent_pos>-*[0-9|a-z]+)\\+(?P<accent_num_1>[0-9|a-z]+)\\+(?P<accent_num_2>[0-9|a-z]+)', re.M | re.I)

  results = [x.groupdict() for x in re.finditer(lab_pattern, label)]
  min_results_len = 3

  if len(results) < min_results_len:
    return None

  to_sec = 1e-7
  label_parsed = []

  for result in results:
    begin = int(result['begin']) * to_sec
    end = int(result['end']) * to_sec
    phoneme = result['phoneme'].lower() if result['phoneme'] != 'N' else result['phoneme']

    label_parsed.append({
      'begin': begin,
      'end': end,
      'phoneme': phoneme,
      'accent': None
    })

  sil_end_sec = label_parsed[0]['end']
  sil_begin_sec = label_parsed[-1]['begin']

  if not include_sil:
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
    sil_begin_sec
  )

def gen_dataset(
  lab_file_path: str,
  wav_file_path: str | None,
  sliding_window_len: int,
  fs: float,
  duration_mag_all: float,
  duration_mag_phonemes: dict,
  duration_mag_indices: list,
  f0_envelope_offset: float,
  f0_envelope_mag: float,
  f0_envelope_len: int,
  f0_normalization_max: float,
  volume_mute_threshold: float,
  volume_envelope_len: int,
  is_mono: bool
) -> tuple[
  NDArray,
  NDArray | None,
  NDArray,
  NDArray | None,
  NDArray | None
] | None:
  parsed = parse_label(lab_file_path, is_mono=is_mono)
  if parsed == None:
    return None

  label_parsed, sil_end_sec, sil_begin_sec = parsed

  invalid_value = -1
  num_before    = math.ceil((sliding_window_len - 1) / 2)
  num_after     = math.floor((sliding_window_len - 1) / 2)

  padding_before = [invalid_value for _ in range(num_before)]
  padding_after  = [invalid_value for _ in range(num_after)]

  phonemes = [x['phoneme'] for x in label_parsed]

  phoneme_numbers = np.array(
    [openjtalk_phoneme_list.index(x) / len(openjtalk_phoneme_list) for x in phonemes],
    dtype=np.float16
  )

  durations = np.array(
    [x['end'] - x['begin'] for x in label_parsed],
    dtype=np.float16
  )

  durations_len = len(durations)
  durations_orig = durations.copy()

  for key in duration_mag_phonemes.keys():
    if not key in openjtalk_phoneme_list:
      continue

    index = openjtalk_phoneme_list.index(key)
    number = np.float16(index / len(openjtalk_phoneme_list))
    mag = duration_mag_phonemes[key]

    bool_array = phoneme_numbers == number
    durations[bool_array] *= mag

  for (index, mag) in duration_mag_indices:
    durations[index] *= mag

  durations *= duration_mag_all

  phoneme_number_segments = seq2seg(
    np.concatenate((padding_before, phoneme_numbers, padding_after)),
    sliding_window_len,
    1
  )

  duration_segments = durations.reshape((-1, 1))

  if wav_file_path == None:
    f0_segments = None
    volume_segments = None

    if is_mono:
      accent_segments = None
    else:
      accents = [x['accent'] for x in label_parsed]
      accent_array = np.array(padding_before + accents + padding_after, dtype=np.float16)
      accent_segments = seq2seg(accent_array, sliding_window_len, 1)
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

    sil_end_index = int(fs * sil_end_sec)
    sil_begin_index = int(fs * sil_begin_sec)

    wave = wave[sil_end_index:sil_begin_index]

    f0_envelope = detect_f0(wave, fs)
    f0_envelope = interp_zeros(f0_envelope)
    f0_envelope *= f0_envelope_mag
    f0_envelope += f0_envelope_offset
    f0_envelope /= f0_normalization_max

    f0_envelope_min = np.min(f0_envelope)
    f0_envelope_max = np.max(f0_envelope)
    f0_envelope_center = f0_envelope_min + ((f0_envelope_max - f0_envelope_min) / 2)

    f0_envelope = resample(f0_envelope, len(wave))

    volume_envelope = detect_volume(wave, fs)

    prev_end = 0
    accents = []
    f0_segments = np.zeros((durations_len, f0_envelope_len), dtype=np.float16)
    volume_segments = np.zeros((durations_len, volume_envelope_len), dtype=np.float16)

    for i, duration in enumerate(durations_orig):
      begin_sec = prev_end
      end_sec = begin_sec + duration

      begin_index = int(begin_sec * fs)
      end_index = int(end_sec * fs)

      f0_segment = f0_envelope[begin_index:end_index]
      f0_segment = resample(f0_segment, f0_envelope_len)
      f0_segments[i] = f0_segment

      accent = 1 if np.average(f0_segment[f0_segment != 0]) >= f0_envelope_center else 0
      accents.append(accent)

      volume_segment = volume_envelope[begin_index:end_index]
      volume_segment = volume_envelope[volume_envelope >= volume_mute_threshold]
      volume_segment = np.zeros(1, dtype=np.float16) if len(volume_segment) <= 0 else volume_segment
      volume_segment = resample(volume_segment, volume_envelope_len)
      volume_segments[i] = volume_segment

      prev_end = end_sec

    accent_array = np.array(padding_before + accents + padding_after, dtype=np.float16)
    accent_segments = seq2seg(accent_array, sliding_window_len, 1)

  return (
    phoneme_number_segments,
    accent_segments,
    duration_segments,
    f0_segments,
    volume_segments
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

  phoneme_number_segments_list = []
  accent_segments_list         = []
  duration_segments_list       = []
  f0_segments_list             = []
  volume_segments_list         = []

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
      config['duration_mag_indices'],
      config['f0_envelope_offset'],
      config['f0_envelope_mag'],
      config['f0_envelope_len'],
      config['f0_normalization_max'],
      config['volume_mute_threshold'],
      config['volume_envelope_len'],
      is_mono=True
    )

    if (dataset is None) or any(x is None for x in dataset):
      continue

    (
      phoneme_number_segments,
      accent_segments,
      duration_segments,
      f0_segments,
      volume_segments
    ) = dataset

    phoneme_number_segments_list += list(phoneme_number_segments)
    accent_segments_list         += list(accent_segments)
    duration_segments_list       += list(duration_segments)
    f0_segments_list             += list(f0_segments)
    volume_segments_list         += list(volume_segments)

  phoneme_number_segments = np.array(phoneme_number_segments_list, dtype=np.float16)
  accent_segments         = np.array(accent_segments_list, dtype=np.float16)
  duration_segments       = np.array(duration_segments_list, dtype=np.float16)
  f0_segments             = np.array(f0_segments_list, dtype=np.float16)
  volume_segments         = np.array(volume_segments_list, dtype=np.float16)

  return {
    'phoneme_number': phoneme_number_segments,
    'accent':         accent_segments,
    'duration':       duration_segments,
    'f0':             f0_segments,
    'volume':         volume_segments
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

def get_dirname(
  path: str
) -> str:
  return os.path.dirname(path)
