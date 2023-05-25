#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=16000
n_fft=1024
n_shift=256

opts=
if [ "${fs}" -eq 16000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=train-clean-100
valid_set=dev-clean
test_sets="dev-clean test-clean"

train_config=conf/tuning/train_jets.yaml
inference_config=conf/tuning/decode_jets.yaml

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en
local_data_opts="--trim_all_silence true" # trim all silence in the audio

./tts_train.sh \
    --ngpu 1 \
    --lang en \
    --feats_type raw \
    --local_data_opts "${local_data_opts}" \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --token_type phn \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --tts_task gan_tts \
    --min_wav_duration 1 \
    --use_sid true \
    ${opts} "$@"
