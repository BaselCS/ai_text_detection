mkdir -p ./nltk_data/tokenizers

wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip -P ./nltk_data/tokenizers/
wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip -P ./nltk_data/tokenizers/

unzip ./nltk_data/tokenizers/punkt.zip -d ./nltk_data/tokenizers/
unzip ./nltk_data/tokenizers/punkt_tab.zip -d ./nltk_data/tokenizers/

rm ./nltk_data/tokenizers/*.zip