import numpy as np 
import operator
def get_DataSet():
	data=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	label=['A','A','B','B']
	return data,label

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


if __name__=='__main__':
	data,label=get_DataSet()
	print(classify0([0,0],data,label,3))

