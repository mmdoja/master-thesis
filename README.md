# Inferring instrument tune from visual data


### Prerequisites

```bash
pip3 install -r requirements.txt
```

Datasets

The link to the datasets are : 
- URMP: [here](http://www2.ece.rochester.edu/projects/air/projects/URMP.html)
- MUSIC: [here](https://github.com/roudimit/MUSIC_dataset)
- Piano: [At Your Fingertips: Automatic Piano Fingering Detection](https://openreview.net/forum?id=H1MOqeHYvB). The dataset from [here](https://drive.google.com/file/d/1kDPZSA7ppOaup9Q1Dab7bW4OXNh9mAQA/view)

To extract poses from videos: 
Use OpenPose API from [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

To extract MIDI from the audio
Use Lunarverus from [here](https://www.lunaverus.com/)

Once you have poses and midi from the videos and audio,
make a 'data' folder such that for each dataset there is a folder for midi and pose for each instrument
```
-data
  -URMP
    -midi
      -cello
      -violin
      -...
    -pose
      -cello
      -violin
      -...
  -MUSIC
    -midi
      -...
    -pose
      -...
```

### Training

For URMP
```bash
CUDA_VISIBLE_DEVICES=6 python train.py -c config/URMP/violin.conf -e exps/urmp-vn
```

For AtinPiano
```bash
CUDA_VISIBLE_DEVICES=6 python train.py -c config/AtinPiano.conf -e exps/atinpiano
```

For MUSIC
```bash
CUDA_VISIBLE_DEVICES=6 python train.py -c config/MUSIC/accordion.conf -e exps/music-accordion
```


### Generating MIDI, sounds and videos

For URMP
```bash
VIDEO_PATH=/path/to/video
INSTRUMENT_NAME='Violin'
python test_URMP.py exps/urmp-vn/checkpoint.pth.tar -o exps/urmp-vn/generate -i Violin -v $VIDEO_PATH -i $INSTRUMENT_NAME
```



For AtinPiano
```bash
VIDEO_PATH=/path/to/video
INSTRUMENT_NAME='Acoustic Grand Piano'
python test_AtinPiano_MUSIC.py exps/atinpiano/checkpoint.pth.tar -o exps/atinpiano/generation -v $VIDEO_PATH -i $INSTRUMENT_NAME
```

For MUSIC
```bash
VIDEO_PATH=/path/to/video
INSTRUMENT_NAME='Accordion'
python test_AtinPiano_MUSIC.py exps/music-accordion/checkpoint.pth.tar -o exps/music-accordion/generation -v $VIDEO_PATH -i $INSTRUMENT_NAME
```

Notes:
- Instrument name ($INSTRUMENT_NAME) can be found [here](https://github.com/craffel/pretty-midi/blob/master/pretty_midi/constants.py#L7)

- If you do not have the video file or you want to generate MIDI and audio only, you can add ``-oa`` flag to skip the generation of video.

## Other Info

### Citation

Please cite the following paper if you feel our work useful to your research.

```
@inproceedings{FoleyMusic2020,
  author    = {Chuang Gan and
               Deng Huang and
               Peihao Chen and
               Joshua B. Tenenbaum and
               Antonio Torralba},
  title     = {Foley Music: Learning to Generate Music from Videos},
  booktitle = {ECCV},
  year      = {2020},
}
```
