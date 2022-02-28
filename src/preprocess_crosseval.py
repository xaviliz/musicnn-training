from pathlib import Path
import argparse

from preprocess import process_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('index_file', help='index file')
    parser.add_argument('audio_dir', help='input audio folder')
    parser.add_argument('data_dir', help='output data file')
    parser.add_argument('--feature-type', '-ft', default='musicnn-melspectrogram',
                        choices=[
                            'musicnn-melspectrogram',
                            'vggish-melspectrogram',
                            'musicnn',
                            'vggish',
                            'openl3',
                            'tempocnn',
                            'spleeter',
                            'effnet_b0',
                            'effnet_b0_3M',
                            'yamnet',
                            'jukebox',
                        ],
                        help='input feature type')
    args = parser.parse_args()

    index_file = args.index_file
    audio_dir = Path(args.audio_dir)
    data_dir = Path(args.data_dir)
    feature_type = args.feature_type

    # set audio representations folder
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True, parents=True)
    fw = open(metadata_dir / 'index.tsv', "w")
    fw.write('')
    fw.close()

    # list audios to process: according to 'index_file'
    files_to_convert = []
    f = open(index_file)
    for line in f.readlines():
        id, audio_path = line.strip().split("\t")
        audio_repr = Path(audio_path).with_suffix(".dat")
        tgt = str(data_dir / audio_repr)
        src = str(audio_dir / audio_path)

        files_to_convert.append((id, src, tgt))

    process_files(files_to_convert, data_dir, feature_type=feature_type)
