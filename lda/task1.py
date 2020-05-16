from lib import read_dataset, train_test_split, lda
import numpy as np
data = read_dataset("pp4data/20newsgroups")
features, topics, topic_counts, word_counts = lda(data.X, n_topics=20, alpha=5/20, beta=0.01, n_iters=1)

with open("topicwords.csv", "w") as out:
    for topic, word_count in enumerate(word_counts):
        # print ("Topic %s" % topic)
        row = [word for word, count in word_count.most_common(5)]
        print (",".join(row), file=out)

np.savetxt("lda-features.txt", features)