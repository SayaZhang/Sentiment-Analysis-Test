import nltk
import ssl
import sys

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download(sys.argv[1])
