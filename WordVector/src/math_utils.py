import math
# refer to TS-SS 
# A Hybrid Geometric Approach for Measuring Similarity Level Among Documents and Document Clustering

def Cosine(vec1, vec2) :
    result = InnerProduct(vec1,vec2) / (VectorSize(vec1) * VectorSize(vec2))
    return result

def VectorSize(vec) :
    return math.sqrt(sum(math.pow(v,2) for v in vec))

def InnerProduct(vec1, vec2) :
    return sum(v1*v2 for v1,v2 in zip(vec1,vec2))

def Euclidean(vec1, vec2) :
    return math.sqrt(sum(math.pow((v1-v2),2) for v1,v2 in zip(vec1, vec2)))

def Theta(vec1, vec2) :
    return math.acos(Cosine(vec1,vec2)) + math.radians(10)

def Triangle(vec1, vec2) :
    theta = math.radians(Theta(vec1,vec2))
    return (VectorSize(vec1) * VectorSize(vec2) * math.sin(theta)) / 2

def Magnitude_Difference(vec1, vec2) :
    return abs(VectorSize(vec1) - VectorSize(vec2))

def Sector(vec1, vec2) :
    ED = Euclidean(vec1, vec2)
    MD = Magnitude_Difference(vec1, vec2)
    theta = Theta(vec1, vec2)
    return math.pi * math.pow((ED+MD),2) * theta/360

def TS_SS(vec1, vec2) :
    return Triangle(vec1, vec2) * Sector(vec1, vec2)

if __name__ == "__main__":
    vec1 = [1,2]
    vec2 = [3,4]
    print(TS_SS(vec1,vec2))