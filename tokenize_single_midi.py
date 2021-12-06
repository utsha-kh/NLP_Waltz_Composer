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
#tokenizer = REMIEncoding(pitch_range, beat_res, nb_velocities, additional_tokens)
tokenizer = MIDILikeEncoding(pitch_range, beat_res, nb_velocities, additional_tokens)
midi = MidiFile('data/waltz69-2.mid')

# Converts MIDI to tokens, and back to a MIDI
# write out only right hand midi
tokens = tokenizer.midi_to_tokens(midi)

# tokens[0]  is sequence corresponding to the right hand 
with open("data/waltz69-2.txt", "w") as outfile:
    outfile.write(' '.join(str(tokens[0][i]) for i in range(0, len(tokens[0]))))

# tokens[1] = []

# converted_back_midi = tokenizer.tokens_to_midi(tokens)#, get_midi_programs(midi))
# #converted_back_midi.dump('out_l.mid')

# # Converts just a selected track
# tokenizer.current_midi_metadata = {'time_division': midi.ticks_per_beat, 'tempo_changes': midi.tempo_changes}
# piano_tokens = tokenizer.track_to_tokens(midi.instruments[0])

# # And convert it back (the last arg stands for (program number, is drum))
# converted_back_track, tempo_changes = tokenizer.tokens_to_track(piano_tokens, midi.ticks_per_beat, (0, False))

print("done!")