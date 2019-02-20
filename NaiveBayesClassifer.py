import numpy as np
import math

import sys


#get data from files
stopWordsDict = {}
stopWordsExtractedDict = {}

negTrainingData_old = ""
posTrainingData_old = ""
negTestData_old = ""
posTestData_old = ""

if len(sys.argv) == 5:
    with open(sys.argv[2],'r') as trainingNegFile:
        negTrainingData_old = trainingNegFile.read()

    with open(sys.argv[1],'r') as trainingPosFile:
        posTrainingData_old = trainingPosFile.read()

    with open(sys.argv[3], 'r') as testPosFile:
        posTestData_old = testPosFile.read()

    with open(sys.argv[4], 'r') as testNegFile:
        negTestData_old = testNegFile.read()

with open('stopwords2.txt','r') as stopWordsFile:
    for line in stopWordsFile:
        stopWordsDict[line[:-1]] = True


negTrainingData_old = negTrainingData_old.split("<br /><br />")
posTrainingData_old = posTrainingData_old.split("<br /><br />")

negTestData_old = negTestData_old.split("<br /><br />")
posTestData_old = posTestData_old.split("<br /><br />")


noError = True
while noError:
    try:
        negTrainingData_old.remove("")
    except ValueError:
        noError = False

noError = True
while noError:
    try:
        posTrainingData_old.remove("")
    except ValueError:
        noError = False


def extractUniqueWordsFromReviews(reviewsArr):
    newReviewsArr = []
    dictIndex = {}
    #Number of times word_i appears in review
    wordCountInReview = {}
    #number of times word_i appears in all reviews
    countInAllReviews = {}

    numberOfWords = 0

    countOfReviewsWithWord = {}

    for stopWord in stopWordsDict:
        newWord = extractWordFromString(stopWord)
        stopWordsExtractedDict[newWord] = True
    
    i_neg = 0
    for indx, review in enumerate(reviewsArr):
        review = review.split(" ")
        noiselessReview = []

        #review dict to get tf-idf feature
        reviewDict = {}
        
        for word in review:
            word = word.lower()
            newWord = extractWordFromString(word)
            
            if newWord != '':
                try:
                    isStopWord = stopWordsExtractedDict[newWord]
                    continue
                except KeyError:

                    numberOfWords = numberOfWords + 1
                    noiselessReview.append(newWord)
                    try:
                        #word already in dictionary
                        isDuplicate = dictIndex[newWord]
                        
                    except KeyError:
                        #word is not in the dictionary add it and increment amount
                        dictIndex[newWord] = i_neg
                        i_neg = i_neg + 1

                    
                    try:
                        #word is duplicate in the review
                        isWordDuplicateInReview = reviewDict[newWord]
                    except KeyError:
                        reviewDict[newWord] = 1

                    try:
                        
                        #increment the number of times the word is seen in all reviews
                        isWordDuplicateInReview = countInAllReviews[newWord]
                        countInAllReviews[newWord] = countInAllReviews[newWord] + 1
                    except KeyError:
                        #initialize the word to 1
                        countInAllReviews[newWord] = 1

                              
        #number of reviews with word_i
        for index,word in enumerate(reviewDict):
            
            try:
                #word has been seen in other reviews
                isWordDuplicateInCount = countOfReviewsWithWord[word]
                countOfReviewsWithWord[word] = countOfReviewsWithWord[word] + 1
            except KeyError:
                #first time the word is being accounted for
                countOfReviewsWithWord[word] = 1

##            print("\n\n\n\n\n")
##            print("Reviews")
##            print(reviewsArr[:2])
##            print("\n\n")
##            print("BOW Dictionary Count ")
##            print(bowDictCount)
##            return (bowDict,bowDictCount)     
                        
        #reviewsArr[indx] = noiselessReview


        

        if(len(noiselessReview) > 0):
            newReviewsArr.append(noiselessReview)

    for indx in range(len(newReviewsArr)):
        for word in newReviewsArr[indx]:
            try:
                
                isInReview = wordCountInReview[str(indx) + word]
                wordCountInReview[str(indx)+ word] = wordCountInReview[str(indx)+ word] + 1
            except KeyError:
                #print(newWord)
                wordCountInReview[str(indx) + word] = 1
            
        
    return (dictIndex,wordCountInReview,countInAllReviews,numberOfWords,countOfReviewsWithWord,newReviewsArr )


def extractWordFromString(string):
    #assumes the string is lower case
    alphabet = {'a':True,'b':True,'c':True,'d':True,'e':True,'f':True,
                'g':True,'h':True,'i':True,'j':True,'k':True,'l':True,
                'm':True,'n':True,'o':True,'p':True,'q':True,'r':True,
                's':True,'t':True,'u':True,'v':True,'w':True,'x':True,
                'y':True,'z':True,
                '0':True,'1':True,'2':True,'3':True,
                '4':True,'5':True,'6':True,'7':True,'8':True,'9':True}
    word = ''
    
    for character in string:
        try:
            #character part of the alphabet
            tempChar = alphabet[character]
            word = word + character
        except:
            #character is not part of the alphabet
            continue
        
    #look for html <br>
    if len(word) >= 2:
        
        if word[-2:] == 'br':
            word = word[:-2]
            
        
    return word






def createVocabulary(positiveReviewTokens, negativeReviewTokens):
    #positiveRevies and negativeReviews are dictionaries with words as keys and the count as values
    vocab = {}
    totalWords = 0

    #add positive review tokens to vocabulary
    for token in positiveReviewTokens:
        try:
            #check if the token is also in negative reviews and add their counts 
            countInNegative = negativeReviewTokens[token]
            countInPositive = positiveReviewTokens[token]
            total = countInNegative + countInPositive
            vocab[token] = total
            totalWords = totalWords + total
                    
        except KeyError:
            #if not add it to vocab with count
            vocab[token] = positiveReviewTokens[token]
            totalWords = totalWords + positiveReviewTokens[token]
            
            

    #add any words that did not show up in positive reviews to vocab
    for token in negativeReviewTokens:
            try:
                countInPositive = vocab[token]
                
            except KeyError:
                vocab[token] = negativeReviewTokens[token]
                totalWords = totalWords + negativeReviewTokens[token]
                
    return (vocab,totalWords)           
    

def ExtractBOWTFIDFFeatures(reviews,wordIndex,wordsInReview,reviewsWithWord):
    
    matrix = np.zeros((len(reviews), len(wordIndex)))
    matrix_tfidf = np.zeros((len(reviews), len(wordIndex)))
    infoArr = []
    tfidfArr = []
    for index in range(len(reviews)):
        tempReviewDict = {}
        totalNumWordsInReview = len(reviews[index])

        for word in reviews[index]:
            try:
                isWordDuplicate = tempReviewDict[word]
                continue

            except KeyError:
                #bow
                column = wordIndex[word]
                numberOfWordsInReview = wordsInReview[str(index) + word]
                matrix[index][column] = numberOfWordsInReview
                tempReviewDict[word] = True

                #tf-idf
                #tf = numberof times it appears in review/total number of words in review
                tf = float(numberOfWordsInReview) / float(totalNumWordsInReview)
                #idf = log(Total number of reviews / Number of reviews with word_i in it)
                idf = math.log(float(len(reviews))/float(reviewsWithWord[word]))
                tf_idf = tf*idf

                matrix_tfidf[index][column] = tf_idf

    
    for i in range(len(wordIndex)):
        infoDict = {}
        infoDict_tfidf = {}
        column = matrix[:,i]

        mean = column.mean()
        std = column.std()
        infoDict["mean"] = mean
        infoDict["std"] = std
        infoArr.append(infoDict)

        column_tfidf = matrix_tfidf[:,i]
        mean = column_tfidf.mean()
        std = column_tfidf.std()
        infoDict_tfidf["mean"] = mean
        infoDict_tfidf["std"] = std
        tfidfArr.append(infoDict_tfidf)


    return (infoArr,tfidfArr)
            
        
def ClassifyReviewsWithBOWFeatures(reviews,negativeFeatures,positiveFeatures,
                                   negativeIndex, positiveIndex):

    newReviews = []
    positiveClassifications = 0
    negativeClassifications = 0
    unclassified = 0
    unclassifiedIndx = []
    
    for review in reviews:
        tempDict = {}
        for word in review:
            try:
                count = tempDict[word]
                tempDict[word] = tempDict[word] + 1

            except KeyError:
                tempDict[word] = 1
        newReviews.append(tempDict)

    for indx,review in enumerate(newReviews):
        temp_pos = 0.0
        temp_neg = 0.0
        

        for word in review:
            count = review[word]
            try:
                index = negativeIndex[word]
                features = negativeFeatures[index]

                mean = features["mean"]
                sigma = features["std"]


                if(sigma == 0):
                    print("Sigma === 0", sigma)
                    print("Word ", word)
                    print("Review ", review)

                seg1 = 1.0 / math.sqrt(2.0*math.pi*math.pow(sigma,2.0))     
                seg2 = math.exp(-(math.pow(float(count) - mean,2)/(2.0*math.pow(sigma,2.0))))

                prob = seg1 * seg2
                temp_neg = temp_neg + prob

                
                
            except KeyError:
                k = 0

            try:
                index = positiveIndex[word]
                features = positiveFeatures[index]
                

                mean = features["mean"]
                sigma = features["std"]

                if(sigma == 0):
                    print("Sigma === 0", sigma)
                    print("Word ", word)
                    print("Review ", review)
                    


                seg1 = 1.0 / math.sqrt(2.0*math.pi*math.pow(sigma,2.0))     
                seg2 = math.exp(-(math.pow(float(count) - mean,2.0)/(2.0*math.pow(sigma,2))))

                prob = seg1 * seg2
                temp_pos = temp_pos + prob
            
                    
            except KeyError:
                k = 0
        if(temp_pos > temp_neg):
            positiveClassifications = positiveClassifications + 1
        elif temp_pos < temp_neg:
            negativeClassifications = negativeClassifications + 1
        else:
            unclassified = unclassified + 1
            unclassifiedIndx.append(indx)

    negPercent = (float(negativeClassifications) / float(len(reviews))) * 100
    posPercent = (float(positiveClassifications) / float(len(reviews))) * 100

    print("Unclassified " , unclassified)
    print("Unclassified index array ", unclassifiedIndx)
    
    return (posPercent, negPercent)
                
def ClassifyReviewsWithTFIDFFeatures(reviews,negativeFeatures,positiveFeatures,
                                   negativeIndex, positiveIndex, negReviewsWithWord, 
                                   posReviewsWithWord):

    newReviews = []
    positiveClassifications = 0
    negativeClassifications = 0
    unclassified = 0
    unclassifiedIndx = []
    
    for review in reviews:
        tempDict = {}
        for word in review:
            try:
                count = tempDict[word]
                tempDict[word] = tempDict[word] + 1

            except KeyError:
                tempDict[word] = 1
        newReviews.append(tempDict)


    lengthOfReviews = len(reviews)
    for indx,review in enumerate(newReviews):
        temp_pos = 0.0
        temp_neg = 0.0
        
        reviewLen = len(review)
        for word in review:
            count = review[word]
            try:
                index = negativeIndex[word]
                features = negativeFeatures[index]
                numberOfReviews = negReviewsWithWord[word]
                

                mean = features["mean"]
                sigma = features["std"]

                if sigma == 0:
                    print("sigma == 0", sigma)
                    print("word", word)
                
                tf = count/float(reviewLen)
                idf = math.log(float(numberOfReviews)/float(lengthOfReviews))



                if(tf*idf == 0):
                    print("word  " ,word)
                    print("TF  ", tf)
                    print("IDF  ", idf)
                    

                tfidf = tf * idf

                
                seg1 = 1.0 / math.sqrt(2.0*math.pi*math.pow(sigma,2.0))     
                seg2 = math.exp(-(math.pow(float(tfidf) - mean,2)/(2.0*math.pow(sigma,2.0))))
                prob = seg1 * seg2
                temp_neg = temp_neg + prob

                
                
            except KeyError:
                k = 0

            try:
                index = positiveIndex[word]
                features = positiveFeatures[index]
                
                
                
                numberOfReviews = posReviewsWithWord[word]

                mean = features["mean"]
                sigma = features["std"]

                if sigma == 0:
                    print("sigma == 0", sigma)
                    print("word", word)

                tf = count/float(reviewLen)
                idf = math.log(float(numberOfReviews)/float(lengthOfReviews))


                tfidf = tf * idf

                if(tf*idf == 0):
                    print("word  ", word)
                    print("TF  ", tf)
                    print("IDF  ", idf)
                    

                seg1 = 1.0 / math.sqrt(2.0*math.pi*math.pow(sigma,float(2)))     
                seg2 = math.exp(-(math.pow(float(tfidf) - mean,2)/(2.0*math.pow(sigma,2))))

                prob = seg1 * seg2
                temp_pos = temp_pos + prob
            
                    
            except KeyError:
                k = 0

        if(temp_pos > temp_neg):
            positiveClassifications = positiveClassifications + 1
        elif temp_pos < temp_neg:
            negativeClassifications = negativeClassifications + 1
        else:
            unclassified = unclassified + 1
            unclassifiedIndx.append(indx)

    negPercent = (float(negativeClassifications) / float(len(reviews))) * 100
    posPercent = (float(positiveClassifications) / float(len(reviews))) * 100

    print("Unclassified " , unclassified)
    print("Unclassified index array ", unclassifiedIndx)
    
    return (posPercent, negPercent) 

    
    

def MultiVariativeBayesClassifierBOW(reviews,countInAllReviewsPos, countInAllReviewsNeg,
                       numberOfPositiveWords, numberOfNegativeWords,vocabLength):

    negClassifications = 0
    posClassifications = 0

    for review_index in range(len(reviews)):
        
        prob_positive = 0.0
        prob_negative = 0.0
        alpha = 0.001

        
        for word in reviews[review_index]:
            occurencesInPositive = 0
            occurencesInNegative = 0

            try:
                occurencesInNegative = countInAllReviewsNeg[word]
            except KeyError:
                k = 0

            try:
                occurencesInPositive = countInAllReviewsPos[word]
            except KeyError:
                k = 0

            numerator_neg = float(occurencesInNegative) + alpha
            numerator_pos = float(occurencesInPositive) + alpha

            denominator_neg = float(numberOfNegativeWords) + float(alpha*float(vocabLength))
            denominator_pos = float(numberOfPositiveWords) + float(alpha*float(vocabLength))

            temp_prob_pos = math.log(numerator_pos/denominator_pos)
            temp_prob_neg = math.log(numerator_neg/denominator_neg)

            prob_positive = prob_positive + temp_prob_pos
            prob_negative = prob_negative + temp_prob_neg
                    
                
        if prob_negative > prob_positive:
            negClassifications = negClassifications + 1
        elif prob_negative < prob_positive:
            posClassifications = posClassifications + 1
        else:
            print("Both negative and positive numbers are the same", prob_negative, prob_positive)


    percentageNeg = float(negClassifications) / float(len(reviews)) * 100
    percentagePos = float(posClassifications) / float(len(reviews)) * 100

    print("Negative and positive classifications", negClassifications, posClassifications)
    return (percentageNeg, percentagePos)
    




def main():
    # a dictionary with word as key and index of the word in matrix as value
    positiveIndex = {}
    negativeIndex = {}
    testNegIndex = {}
    testPosIndex = {}
    
    
    #dictionaries with word as key and word's occurence in a review as value
    posCountInReview = {}
    negCountInReview= {}
    testPosCountInReview = {}
    testNegCountInReview= {}

    #dictionaries with word as key and words's occurrence in the whole class
    positiveCountInAll = {}
    negativeCountInAll = {}
    testPosCountInAll = {}
    testNegCountInAll = {}

    
    #list of reviews with each review being a list of words/token
    negTrainingData = []
    posTrainingData = []


    posReviewsWithWord = {}
    negReviewsWithWord = {}
    
    vocabuary = {}

    totalWords  = 0
    numberOfPositiveWords = 0
    numberOfNegativeWords = 0

    #print("Original Negative Review")
    #print(negTrainingData[4357])

##    print("Original Positive Review")
##    print(posTrainingData[3640])

    #get information of negative & positive reviews


    
    (negativeIndex,negCountInReview,
     negativeCountInAll, numberOfNegativeWords,
     negReviewsWithWord, negTrainingData) = extractUniqueWordsFromReviews(negTrainingData_old)

    (positiveIndex,posCountInReview,
     positiveCountInAll, numberOfPositiveWords,
     posReviewsWithWord, posTrainingData) = extractUniqueWordsFromReviews(posTrainingData_old)

    
    
    
    

    (vocabulary,totalWords) = createVocabulary(positiveCountInAll,negativeCountInAll)

    #get information for test data
    (testPosIndex, testPosCountInReview,
     testPosCountInAll, numberOfTestPosWords,
     posTestReviewsWithWord, posTestData) = extractUniqueWordsFromReviews(posTestData_old)

    
    (testNegIndex, testNegCountInReview,
     testNegCountInAll,numberOfTestNegWords,
     negTestReviewsWithWord, negTestData) =  extractUniqueWordsFromReviews(negTestData_old)


    
    print("Bayes function with POSITVE test data")
    (percentageNeg, percentagePos) = MultiVariativeBayesClassifierBOW(posTestData, positiveCountInAll,
                                                        negativeCountInAll,numberOfPositiveWords,
                                                        numberOfNegativeWords, totalWords)
    print("Negative Percentage", percentageNeg)
    print("Positive Percentage", percentagePos)
    print("\n\n")
    

    print("Bayes function with NEGATIVE test data")
    (percentageNeg, percentagePos) = MultiVariativeBayesClassifierBOW(negTestData, positiveCountInAll,
                                                        negativeCountInAll,numberOfPositiveWords,
                                                        numberOfNegativeWords, totalWords)
    print("Negative Percentage", percentageNeg)
    print("Positive Percentage", percentagePos)
    print("\n\n")
    

    
    
    (negativeFeaturesArr,negTFIDFFeaturesArr) = ExtractBOWTFIDFFeatures(negTrainingData,negativeIndex,negCountInReview, negReviewsWithWord)
    
    (positiveFeaturesArr,posTFIDFFeaturesArr) = ExtractBOWTFIDFFeatures(posTrainingData,positiveIndex,posCountInReview, posReviewsWithWord)


    print("Classifying NEGATIVE Test Data with Guassian BOW")
    (posPercentage, negPercentage) = ClassifyReviewsWithBOWFeatures(negTestData,negativeFeaturesArr,
                                                                    positiveFeaturesArr, negativeIndex,
                                                                    positiveIndex)
    print("Negative Percentage", negPercentage)
    print("Positive Percentage", posPercentage)
    print("\n\n")



    print("Classifying POSITIVE Test Data with Guassian BOW")
    (posPercentage, negPercentage) = ClassifyReviewsWithBOWFeatures(posTestData,negativeFeaturesArr,
                                                                    positiveFeaturesArr, negativeIndex,
                                                                    positiveIndex)
    print("Negative Percentage", negPercentage)
    print("Positive Percentage", posPercentage)
    print("\n\n")
   
        

    print("Classifying NEGATIVE Test Data with Guassian TFIDF")
    (posPercentage, negPercentage) = ClassifyReviewsWithTFIDFFeatures(negTestData,negTFIDFFeaturesArr,posTFIDFFeaturesArr,
                                                                    negativeIndex, positiveIndex, negReviewsWithWord, 
                                                                    posReviewsWithWord)

    print("Negative Percentage", negPercentage)
    print("Positive Percentage", posPercentage)
    print("\n\n")


    print("Classifying POSITIVE Test Data with Guassian TFIDF")
    (posPercentage, negPercentage) = ClassifyReviewsWithTFIDFFeatures(posTestData,negTFIDFFeaturesArr,posTFIDFFeaturesArr,
                                                                    negativeIndex, positiveIndex, negReviewsWithWord, 
                                                                    posReviewsWithWord)

    print("Negative Percentage", negPercentage)
    print("Positive Percentage", posPercentage)
    print("\n\n")
    


    
main()







    


        
        
    



        
    
    





