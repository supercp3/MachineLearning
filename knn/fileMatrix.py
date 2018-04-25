import numpy as np
import matplotlib.pyplot as plt
import operator

#读取文件，将文件内容转化为矩阵，返回结果为样本矩阵和标签列表
def fileMatrix(filename):
	fr=open(filename)
	lines=fr.readlines()
	numlines=len(lines)
	mat=np.zeros((numlines,3))
	classLabelVector=[]
	index=0
	for line in lines:
		line=line.strip()
		listFromLine=line.split('\t')
		mat[index,:]=listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index+=1
	return mat,classLabelVector

#将特征值归一化到0-1区间内
def autoNorm(dataSet):
	minVals=dataSet.min(0)
	maxVals=dataSet.max(0)
	ranges=maxVals-minVals
	normDateSet=np.zeros(np.shape(dataSet))
	m=dataSet.shape[0]
	normDateSet=dataSet-np.tile(minVals,(m,1))
	normDateSet=normDateSet/np.tile(ranges,(m,1))
	return normDateSet,ranges,minVals

#分类函数，计算输入的样本的预测类别
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

#数据分为训练数据和测试数据并且进行实验对比，输出错误率
def datingClassTest():
	ratio=0.1
	datingMat,datingLabels=fileMatrix('datingTestSet2.txt')
	normMat,ranges,minVals=autoNorm(datingMat)
	m=normMat.shape[0]
	numTestVecs=int(m*ratio)
	errorCount=0.0
	for i in range(numTestVecs):
		classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)
		print("the classifier came back with:%d,the real answer is:%d"%(classifierResult,datingLabels[i]))
		if(classifierResult!=datingLabels[i]):
			errorCount+=1.0 
	print(errorCount)
	print("the total error rate is:%f"%(errorCount/float(numTestVecs)))
#构建完整的分类系统
def classifyPerson():
	while 1:
		resultList=['not at all','in small doses','in large doses']
		percentTats=float(input("1.input your percentage of time spent playng video games:"))
		ffMiles=float(input("2.input frequent flier miles earned per year:"))
		iceCream=float(input("3.input liters of ice cream consumed per year:"))
		datingDataMat,datingLabels=fileMatrix('datingTestSet2.txt')
		normMat,ranges,minVals=autoNorm(datingDataMat)
		inArr=np.array([ffMiles,percentTats,iceCream])
		classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
		print("Dear sir/madam,you will probably like this person:\n",resultList[classifierResult-1])

if __name__ == '__main__':
	classifyPerson()
	#datingClassTest()
	'''
	filename='datingTestSet2.txt'
	data,label=fileMatrix(filename)
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.scatter(data[:,0],data[:,1],15*np.array(label),15*np.array(label))
	plt.show()
	normData,ranges,minVals=autoNorm(data)
	print(normData)
	'''
