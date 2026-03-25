from math import log
import operator

def calcShannonEnt(dataSet):
    """计算熵(用于C4.5算法)"""
    numEntries = len(dataSet)
    if numEntries == 0: return 0
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def calcGini(dataSet):
    """计算基尼指数(用于CART算法)"""
    numEntries = len(dataSet)
    if numEntries == 0: return 0
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    gini = 1.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        gini -= prob ** 2
    return gini

def splitDataSet(dataSet, axis, value, continuous=False, direction='left'):
    """划分数据集"""
    retDataSet = []
    for featVec in dataSet:
        if continuous:
            if direction == 'left':
                if featVec[axis] <= value:
                    retDataSet.append(featVec) # 连续值不删除列
            else:
                if featVec[axis] > value:
                    retDataSet.append(featVec) # 连续值不删除列
        else:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet, method='C4.5'):
    """选择最佳划分特征"""
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestGainRatio = -1.0
    bestGini = 100000.0
    bestFeature = -1
    bestSplitPoint = None

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = sorted(set(featList))
        
        # 检查是否为连续值
        is_continuous = len(uniqueVals) > 10

        if is_continuous:
            for j in range(len(uniqueVals) - 1):
                splitPoint = (uniqueVals[j] + uniqueVals[j+1]) / 2.0
                subLeft = [ex for ex in dataSet if ex[i] <= splitPoint]
                subRight = [ex for ex in dataSet if ex[i] > splitPoint]
                probLeft = len(subLeft) / float(len(dataSet))
                probRight = len(subRight) / float(len(dataSet))
                
                if method == 'C4.5':
                    newEntropy = probLeft * calcShannonEnt(subLeft) + probRight * calcShannonEnt(subRight)
                    infoGain = baseEntropy - newEntropy
                    splitInfo = 0.0
                    if probLeft > 0: splitInfo -= probLeft * log(probLeft, 2)
                    if probRight > 0: splitInfo -= probRight * log(probRight, 2)
                    gainRatio = infoGain / splitInfo if splitInfo != 0 else 0
                    if gainRatio > bestGainRatio:
                        bestGainRatio = gainRatio
                        bestFeature = i
                        bestSplitPoint = splitPoint
                elif method == 'CART':
                    gini = probLeft * calcGini(subLeft) + probRight * calcGini(subRight)
                    if gini < bestGini:
                        bestGini = gini
                        bestFeature = i
                        bestSplitPoint = splitPoint
        else:
            if method == 'C4.5':
                newEntropy = 0.0
                splitInfo = 0.0
                for value in uniqueVals:
                    subDataSet = [ex for ex in dataSet if ex[i] == value]
                    prob = len(subDataSet) / float(len(dataSet))
                    newEntropy += prob * calcShannonEnt(subDataSet)
                    if prob > 0: splitInfo -= prob * log(prob, 2)
                infoGain = baseEntropy - newEntropy
                gainRatio = infoGain / splitInfo if splitInfo != 0 else 0
                if gainRatio > bestGainRatio:
                    bestGainRatio = gainRatio
                    bestFeature = i
                    bestSplitPoint = None
            elif method == 'CART':
                gini = 0.0
                for value in uniqueVals:
                    subDataSet = [ex for ex in dataSet if ex[i] == value]
                    prob = len(subDataSet) / float(len(dataSet))
                    gini += prob * calcGini(subDataSet)
                if gini < bestGini:
                    bestGini = gini
                    bestFeature = i
                    bestSplitPoint = None

    return bestFeature, bestSplitPoint

def majorityCnt(classList):
    """多数表决函数"""
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels, method='C4.5', depth=0, max_depth=10):
    """递归构建决策树"""
    classList = [example[-1] for example in dataSet]
    # 停止条件 1：类别完全相同
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 停止条件 2：没有更多特征或达到最大深度
    if len(dataSet[0]) == 1 or depth >= max_depth:
        return majorityCnt(classList)
    
    bestFeat, bestSplitPoint = chooseBestFeatureToSplit(dataSet, method)
    if bestFeat == -1:
        return majorityCnt(classList)
        
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    
    if bestSplitPoint is not None:
        # 连续值：不删除特征
        subLabels = labels[:]
        leftName = f"<= {bestSplitPoint:.4f}"
        rightName = f"> {bestSplitPoint:.4f}"
        
        subLeft = splitDataSet(dataSet, bestFeat, bestSplitPoint, continuous=True, direction='left')
        subRight = splitDataSet(dataSet, bestFeat, bestSplitPoint, continuous=True, direction='right')
        
        # 防止无限递归
        if not subLeft or not subRight:
            return majorityCnt(classList)
            
        myTree[bestFeatLabel][leftName] = createTree(subLeft, subLabels, method, depth+1, max_depth)
        myTree[bestFeatLabel][rightName] = createTree(subRight, subLabels, method, depth+1, max_depth)
    else:
        # 离散值：删除特征
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        newLabels = labels[:bestFeat] + labels[bestFeat+1:]
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, bestFeat, value, continuous=False)
            myTree[bestFeatLabel][value] = createTree(subDataSet, newLabels, method, depth+1, max_depth)
            
    return myTree

def classify(inputTree, featLabels, testVec):
    """使用决策树进行分类"""
    if not isinstance(inputTree, dict):
        return inputTree
    
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    
    try:
        featIndex = featLabels.index(firstStr)
    except ValueError:
        return None
        
    testVal = testVec[featIndex]
    
    # 判断是否为连续值分支
    is_continuous = False
    keys = list(secondDict.keys())
    for k in keys:
        if isinstance(k, str) and (k.startswith('<=') or k.startswith('>')):
            is_continuous = True
            break
            
    if is_continuous:
        # 提取划分点
        splitPoint = float(keys[0].split(' ')[1])
        if testVal <= splitPoint:
            key = [k for k in keys if k.startswith('<=')][0]
        else:
            key = [k for k in keys if k.startswith('>')][0]
        return classify(secondDict[key], featLabels, testVec)
    else:
        if testVal in secondDict:
            return classify(secondDict[testVal], featLabels, testVec)
        else:
            # 遇到未知特征值，返回当前子树下最多的类
            return classify(secondDict[keys[0]], featLabels, testVec)
