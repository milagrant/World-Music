audiolist = open("data/audiolist_edited.txt", "r")
harmony = open("data/csvlist_harmony.txt", "w")
melody = open("data/csvlist_melody.txt", "w")
timbre = open("data/csvlist_timbre.txt", "w")
rhythm = open("data/csvlist_rhythm.txt", "w")

for line in audiolist:
    filename = line[6:].replace('mp3', 'csv')
    harmony.write("csvfiles/harmony/" + filename)
    melody.write("csvfiles/timbre/" + filename)
    timbre.write("csvfiles/timbre/" + filename)
    rhythm.write("csvfiles/rhythm/" + filename)
