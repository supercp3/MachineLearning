import numpy as np
import os
import operator
def img2Vector(filename):
	returnVect=np.zeros((1,1024))
	fr=open(filename)
	for i in range(32):
		lineStr=fr.readline()
		for j in range(32):
			returnVect[0,32*i+j]=int(lineStr[j])
	return returnVect
#x=img2Vector('digits/trainingDigits/0_0.txt')

def classify0(intx,dataset,labels,k):
	size=dataset.shape[0]
	#将intx的数组复制size次，然后与原数组进行减法运算
	mat=np.tile(intx,(size,1))-dataset
	sgmat=mat**2
	sgdistance=sgmat.sum(axis=1)
	distances=sgdistance**0.5
	sortedDistIndex=distances.argsort()
	classCount={}
	for i in range(k):
		vlabel=labels[sortedDistIndex[i]]
		classCount[vlabel]=classCount.get(vlabel,0)+1 
	sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def handWritingClassTest():
	hwLabels=[]
	trainingFileList=os.listdir('digits/trainingDigits')
	m=len(trainingFileList)
	trainingMat=np.zeros((m,1024))
	for i in range(m):
		fileNameStr=trainingFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:]=img2Vector('digits/trainingDigits/%s'%fileNameStr)
	testFileList=os.listdir('digits/testDigits')
	errorCount=0.0 
	mTest=len(testFileList)
	for i in range(mTest):
		fileNameStr=testFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		vectorUnderTest=img2Vector('digits/testDigits/%s'%fileNameStr)
		classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
		print("the classifier came back with:%d,the real answer is:%d"%(classifierResult,classNumStr))
		if(classifierResult!=classNumStr):
			errorCount+=1.0 
	print("the total number of errors is:%d"%errorCount)
	print("the total error rate is:%f"%(errorCount/float(mTest)))
handWritingClassTest()