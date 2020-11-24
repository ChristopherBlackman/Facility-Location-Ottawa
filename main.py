import re
import numpy as np
from numpy import linalg as la
import itertools as it
from pprint import pprint
from sklearn.neighbors import NearestNeighbors
import multiprocessing
KML_ROAD_FILE = "./data/Road_Centrelines.kml"
DOT_FILE = "./data/test.dot"


def main():
    #print("Extracting Data")
    #pre_data = organize_data()
    #print("Data Extracted")

    #print("Converting To PointSet")
    #point_set_x, point_set_y, edge_set_x, edge_set_y = to_points(pre_data)


    #print("Cal. Intersections")
    #point_intersections(pre_data,point_set_x,point_set_y)

    #print("Seg. Intersections")
    #segment_intersections(pre_data,edge_set_x,edge_set_y)

    test_set()

def test_set():
    print("Extracting data from : ",DOT_FILE)
    vertex_set, edge_set = dot_extract()

def dot_extract(norm=np.linalg.norm):
    with open(DOT_FILE,'rt',encoding='utf-8') as f:
        vertex = re.compile("\w+\s\[pos=\"\d+,\d+\"\];")
        edge = re.compile("\w+\s--\s\w+;")
        
        vertex_dict = {}
        edge_dict = {}

        for line in f.readlines():
            l = line.strip()
            if vertex.search(l):
                pos = [ float(i) for i in re.findall("\d+",l)]
                v = re.findall("\A\w+",l)
                vertex_dict[v[0]] = np.array(pos)
            if edge.search(l):
                e = re.findall("\w+",l)
                if edge_dict.get(e[0],None) is None:
                    edge_dict[e[0]] = {}
                edge_dict[e[0]][e[1]] = 0
                if edge_dict.get(e[1],None) is None:
                    edge_dict[e[1]] = {}
                edge_dict[e[1]][e[0]] = 0

                    

        for v1, v_to in edge_dict.items():
            for v2 in v_to:
                dist = norm(vertex_dict[v1]-vertex_dict[v2]) 
                edge_dict[v1][v2] = dist
    return vertex_dict, edge_dict

class KMeans:
    def __init__(self,n_clusters=2, random_state=0,n_init=10,max_iter=300)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

    # compute k-means, and find clusters
    # V : Vertex Set -> Pos : [x,y]
    # E : Edge Set -> Weights : float
    def fit(V,E):
        # clamp two random verticies as centers
        

def to_points(pre_data):
    tmp_x = []
    tmp_y = []
    
    for road in pre_data:
        for point in road['coordinates']:
            tmp_x.append(point)
            tmp_y.append(road['object_id'])


    edge = []
    edge_id = []
    for i in range(len(tmp_x)-1):
        if tmp_y[i] == tmp_y[i+1]:
            edge.append([tmp_x[i],tmp_x[i+1]])
            edge_id.append(tmp_y[i])

    return (np.array(tmp_x),np.array(tmp_y),np.array(edge),np.array(edge_id))

# get the 10 NN of a point
# assumptions : use Euclidian Distance for NN calculation, but test with spherical (curvature is almost plane like)
# assumptions : Nearest Radius is 0.0000000000001 -> least Sig. Digit in longitued, latitutde
# ToDo : Determine Output Data Format for graph
def point_intersections(pre_data,point_set_x,point_set_y):
    #neigh = NearestNeighbors(n_neighbors=10,metric=spherical_dist, radius=0.4) 
    neigh = NearestNeighbors(n_neighbors=10, radius=0.0000000000001) 
    neigh.fit(point_set_x)

    for road in pre_data:
            Y = neigh.kneighbors(road['coordinates'],10,return_distance=False)

            for point, nn in zip(road['coordinates'],Y):
                for index in nn:
                    if (spherical_dist(point,point_set_x[index]) < 10 and  not road['object_id'] == point_set_y[index]):
                        #print(spherical_dist(point,point_set_x[index]))
                        #print(road['object_id'], point_set_y[index])
                        pass
               

#def split(a, n):
    #k, m = divmod(len(a), n)
    #return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def segment_intersections(pre_data,edge_set_x,edge_set_y,p=1):
    i = 0
    t1 = edge_set_x[:,0,:]
    t2 = edge_set_x[:,1,:]

    neigh = NearestNeighbors(n_neighbors=10, radius=0.0000000000001) 
    neigh.fit(point_set_x)
    
    #p = Pool(6)
    #p.map(h_cal_road,zip(pre_data,t1,t2))

    #with multiprocessing.Pool(processes=p) as pool:
    '''
    for road in pre_data:
        h_cal_road(road,t1,t2)
        i = i + 1
        print(i)
        '''

def h_cal_road(road,t1,t2):
        start = road['coordinates'][0]
        end = road['coordinates'][-1]

        cal_length(t1,t2,start)
        cal_length(t1,t2,end)


# 2d only
# A is from t1 to point
# B is from t1 to t2
# returns the projection in terms of how much it is scaled by B
def cal_projections(A,B):
    t = np.apply_along_axis(la.norm,1,B)
    return np.einsum('ij, ij->i',A,(B/t.reshape((t.shape[0],1))))/t

# returns the othogonal length of p to segment
# t1 is segment endpoint
# t2 is segment endpoint
# p  is point we are calculating orthogonal distance
#      p
#     /|
#    / |
#   /  | <-- length returned
#  /   |
#t1-----t2
def cal_length(t1,t2,p):
    l = cal_projections(t1-p,t1-t2)
    f = np.vectorize(lambda_condition)
    l = f(l)
    
    pp = t1*(1-l.reshape((l.shape[0],1))) + t2*(l.reshape((l.shape[0],1)))
    
    return np.apply_along_axis(la.norm,1,pp-p)

# checks if line hits segment
# return x if in range
# return inf if out of range
def lambda_condition(x):
    if x < 0.0 or x > 1.0:
        return np.inf
    return x


# Credit : Jaime
# https://stackoverflow.com/questions/19413259/efficient-way-to-calculate-distance-matrix-given-latitude-and-longitude-data-in
# Note : not matching up with values from ottawa, but may be a good estimate
# Note : note taking into account depth of terrain
def spherical_dist(pos1, pos2, r=6370994):
    pos1 = pos1 * np.pi / 180.0
    pos2 = pos2 * np.pi / 180.0
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

# Not The Right Way to Parse A XML/KML File
# Proper way would be to define a recursive Context Free Grammer, but like, Python (this is faster)
def organize_data():
    data_points = []


    road = None
    count = 0

    with open(KML_ROAD_FILE,'rt',encoding="utf-8") as f:
        btwn = re.compile(">[\w\s,-\.'\(\)]+<")
        asci = re.compile("[\w\s,-\.'\(\)]+")

        for line in f.readlines():
            # check for new placemark
            if re.search("<Placemark>",line):
                road = {'coordinates':None,'object_id':None,'municipality':None,'name':None,'coordinates':[]}
            if re.search("</Placemark>",line):
                data_points.append(road)

            # check for object_id
            if re.search("<SimpleData name=\"OBJECTID\">",line):
                road['object_id'] = asci.search(btwn.search(line).group(0)).group(0)
                pass

            # check for object_id
            if re.search("<SimpleData name=\"ROAD_NAME\">",line):
                road['name'] = asci.search(btwn.search(line).group(0)).group(0)
                pass

            # check for municipality
            if re.search("<SimpleData name=\"MUNICIPALITY\">",line):
                #print(re.search(">[\w\s]+<",line))
                road['municipality'] = asci.search(btwn.search(line).group(0)).group(0)
                pass

            # check for longitute, and latitude of path
            if re.search("<coordinates>",line):
                coordinates = re.findall("-{0,1}\d+\.\d+",line)
                temp = []
                for i in range(int(len(coordinates)/2)):
                    temp.append((float(coordinates[2*i]),float(coordinates[2*i+1])))
                road['coordinates'] = np.array(temp)
                count = count + len(temp)

    return data_points










if __name__ == "__main__":
    print("Running Main File")
    main()
