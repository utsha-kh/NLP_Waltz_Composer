from miditok import MIDILikeEncoding
from miditoolkit import MidiFile

# Our parameters
pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)

# Creates the tokenizer and loads a MIDI
tokenizer = MIDILikeEncoding(pitch_range, beat_res, nb_velocities, additional_tokens)

in_file = open("data/waltz64-2.txt", 'r')
content = in_file.read()
tokens = [[int(content.split(' ')[i]) for i in range(len(content.split(' ')))]]

midi = tokenizer.tokens_to_midi(tokens)
midi.dump("data/waltz64_2_r.mid")

print("done!")