import torch
from FeedforwardNeuralNetwork import FeedforwardNN
from process_data import read_file, clean_data, tokenize, tokenizecorpus, word_to_index_map, idx_to_word,  word_frequency, create_vocab, train_test_split,create_context_and_labels, get_id_word



EMBEDDING_DIM = 200
BATCH_SIZE = 200
EPOCHS = 4
H = 1000
N_GRAMS = 4
PATH = "Brown.txt"


#Reading the file 
corpus = read_file("Brown.txt")

corpus = [clean_data(sentence) for sentence in corpus]


corpus = [tokenize(sentence) for sentence in corpus]


train_corpus, test_corpus = train_test_split(corpus, train_size=0.7)

word_freq_map = word_frequency(train_corpus)

vocabulary = create_vocab(train_corpus, word_freq_map)

word_to_ix = word_to_index_map(vocabulary)

train_corpus = [word for sentence in train_corpus for word in sentence]

test_corpus = [word for sentence in test_corpus for word in sentence]

train_data = create_context_and_labels(train_corpus, N_GRAMS)
#print(train_data)
test_data = create_context_and_labels(test_corpus, N_GRAMS)

VOCAB_SIZE = len(vocabulary)

# BUILDING MODEL
model = FeedforwardNN(embedding_dim=EMBEDDING_DIM, vocab_size=VOCAB_SIZE, context_size=N_GRAMS-1, h=H)
#Defining the Learning rate and optimizer(Using SGD)
LEARNING_RATE = 0.001
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
losses = []
for epoch in range(EPOCHS):
    for i, (context, target) in enumerate(train_data):
        context_ids = list(map(lambda w: get_id_word(w, word_to_ix), context))
        context_vars = torch.LongTensor(context_ids)

        # FORWARD PASS
        output = model(context_vars)
        loss = criterion(output, torch.LongTensor([get_id_word(target, word_to_ix)]))

        # BACKWARD PASS
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        losses.append(loss)

        if (i+1) % 200 == 0:
            print(f"epoch {epoch+1}/{EPOCHS}, step {i+1}/{i+1}, loss = {loss.item():.4f}")
