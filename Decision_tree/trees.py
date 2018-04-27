from math import log
import operator
import treePlotter
'''
计算给定数据集的香浓熵
'''
#计算信息熵
def calcShannonEnt(dataSet):
	numEntries=len(dataSet)
	labelCounts={}
	for featVec in dataSet:
		currentLabel=featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel]+=1
	shannonEnt=0.0
	for key in labelCounts:
		prob=float(labelCounts[key])/numEntries
		shannonEnt-=prob*log(prob,2)
	return shannonEnt

def createDataSet():
	dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
	labels=['no surfacing','flippers']
	return dataSet,labels

#划分数据集
def splitDataSet(dataSet,axis,value):
	retDataSet=[]
	for featVec in dataSet:
		if featVec[axis]==value:
			reducedFeatVec=featVec[0:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures=len(dataSet[0])-1 #计算特征数
	baseEntropy=calcShannonEnt(dataSet)#计算原始信息熵info(D)
	bestInfoGain=0.0 
	bestFeature=-1
	for i in range(numFeatures):#每次进行特征数迭代
		featList=[example[i] for example in dataSet]#选取一个特征列
		uniqueVals=set(featList)#每个特征的不同取值
		newEntropy=0.0 
		for value in uniqueVals:#计算特征列不同取值的信息熵
			subDataSet=splitDataSet(dataSet,i,value)
			prob=len(subDataSet)/float(len(dataSet))
			newEntropy+=prob*calcShannonEnt(subDataSet)
		infoGain=baseEntropy-newEntropy#计算细腻增益
		if infoGain>bestInfoGain:#选择信息增益最大的特征
			bestInfoGain=infoGain
			bestFeature=i 
	return bestFeature
#投票表决代码
def majorityCnt(classList):
	classCount={}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote]=0 
		classCount[vote]+=1
	sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sorteedClassCount[0][0]

def createTree(dataSet,labels):
	classList=[example[-1] for example in dataSet]
	#类别完全相同，则停止划分
	if classList.count(classList[0])==len(classList):
		return classList[0]
	#遍历完所有特征时候返回出现次数最多的类别
	if len(dataSet[0])==1:
		return majorityCnt(classList)

	bestFeat=chooseBestFeatureToSplit(dataSet)
	bestFeatLabel=labels[bestFeat]
	myTree={bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues=[example[bestFeat] for example in dataSet]
	uniqueVals=set(featValues)
	for value in uniqueVals:
		subLabels=labels[:]
		myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
	return myTree 

def classify(inputTree,featLabels,testVec):
	firstStr=list(inputTree.keys())[0]
	secondDict=inputTree[firstStr]
	featIndex=featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex]==key:
			if type(secondDict[key]).__name__=='dict':
				classLabel=classify(secondDict[key],featLabels,testVec)
			else:
				classLabel=secondDict[key]
	return classLabel
'''
myData,labels=createDataSet()
mytree=createTree(myData,labels)
print(mytree)
'''
'''
myData,labels=createDataSet()
print(labels)
myTree=treePlotter.retrieveTree(0)
print(myTree)
print(classify(myTree,labels,[1,0]))
print(classify(myTree,labels,[1,1]))
'''
#使用pickle模块存储决策树
'''
def storeTree(inputTree,filename):
	import pickle
	fw=open(filename,'wb')
	pickle.dump(inputTree,fw)
	fw.close()

def grabTree(filename):
	import pickle
	fr=open(filename,'rb')
	return pickle.load(fr)
myData,labels=createDataSet()
myTree=treePlotter.retrieveTree(0)
storeTree(myTree,'classifierStorage.txt')
grabTree('classifierStorage.txt')
'''