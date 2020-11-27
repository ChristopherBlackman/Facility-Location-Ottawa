import re
import random

import itertools as it

import numpy as np
from numpy import linalg as la

from pprint import pprint

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi

from dijkstar import Graph, find_path

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
    model = KMeans()
    print(model.fit(vertex_set,edge_set))

    model_2 = VoronoiFacilitySelection()
    print(model_2.fit(vertex_set,edge_set))

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


def adj_list_to_graph(E):
    graph = Graph()
    for v1, v_to in E.items():
        for v2 in v_to:
            graph.add_edge(v1,v2,E[v1][v2])
    return graph

class VoronoiFacilitySelection:
    def __init__(self,random_state=42,threshold=1,max_iter=300,n_cells=3):
        self.random_state = random_state
        self.threshold = threshold
        self.max_iter = max_iter
        self.n_cells = n_cells

    # Voronoi Construction On 2 Points
    def Voronoi_2_points(self):
        pass

    # the largest circle inside a voronoi cell 
    def objective_function(self,V,E,CH_points,CH_segments,vor):

        #output : delta from center at voronoin point i (maximum radii)
        distances = np.zeros(vor.points.shape)
            
        print("Objective : Voronoi Centers")
        # compute distances of centers, with voronoi verticies
        for i, point_region in zip(range(len(vor.points)),vor.point_region):
            vor_v_i = vor.regions[point_region]
            points = np.array([vor.vertices[p] for p in vor_v_i if p >= 0])
            center = vor.points[i]
            delta = points - center
            delta_max = delta[np.argmax(np.apply_along_axis(np.linalg.norm,1,delta))]
            
            # check if delta max is bigger than any other distances found thus far for point at index i
            if la.norm(distances[i]) < la.norm(delta_max):
                distances[i] = delta_max


        print("Objective : Points on city border")
        # compute distance of centers, with C.H, points

        # this section will inside the loop
        neigh = NearestNeighbors(n_neighbors=1, radius=0.0000000000001) 
        neigh.fit(vor.points)

        Y = neigh.kneighbors(CH_points,1,return_distance=False) 

        for center_i,CH_p in zip(Y,CH_points):
            center = vor.points[center_i]
            delta = CH_p-center
    
            #check if delta is greater than any other delta of center_i
            if la.norm(distances[center_i]) < la.norm(delta):
                distances[center_i] = delta





        print("Objective : Edge Intersections on city border")

        # compute distance of centers, with edge intersections

        #rotation matrix 90 deg
        rot = np.array([[0,-1],[1,0]])


        #compute centroid
        hull = set()
        for points , voronoi_ridge in vor.ridge_dict.items():
            if voronoi_ridge[np.argmin(voronoi_ridge)] < 0:
                hull.add(points[0])
                hull.add(points[1])

        hull = np.array([vor.points[i] for i in hull])
        centroid = np.sum(hull,axis=0)/hull.shape[0]


        for points_index, voronoi_ridge in vor.ridge_dict.items():


            # find the infnite voronoi ridges
            if voronoi_ridge[np.argmin(voronoi_ridge)] >= 0:
                continue

            p0 = vor.vertices[voronoi_ridge[np.argmax(voronoi_ridge)]]
            p1 = vor.points[points_index[0]]
            p2 = vor.points[points_index[1]]
            
            #bisector
            q = (p1 + p2)/2
            s = (p1-p2)@rot

            t1 = np.sign(np.linalg.det(np.hstack((np.array(([p1,p2,q+s])),np.ones((3,1))))))
            t2 = np.sign(np.linalg.det(np.hstack((np.array(([p1,p2,centroid])),np.ones((3,1))))))
            
            # make sure the the edge is pointing outwards from the convex hull
            if t1 == t2:
                s = -s
                
            #original math proof from :
            #https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
            #
            #take the idea of line segment intersection
            # q+ s   p + r
            # \    /
            #  \  /
            #   \/
            #   /\
            #  /  \
            # /    \
            # p     q
            #
            # p + t*r = q + u*s ; where t and u are scalars
            # (p + t*r)xs = (q + u*s)xs
            # pxs + t*rxs = qxs + 0
            # t*rxs = qxs - pxs
            # t = [(q-p)xs]/rxs
            # if t is negaive, then we went left of p, thus did not intersect
            # if |p + tr - p| > |p + r - p| then we have gone past our endpoint c2
            # note, this is also t|r| > |r| -> t > 1
            
            for c1, c2 in CH_segments:
                p = c1
                r = c2 - c1

                t = np.cross((q-p),s)/np.cross(r,s)
                u = np.cross((q-p),r)/np.cross(r,s)
                

                # ray q intersects segment p at point p + tr (u cannot b zero, else we go backards, t must be btwn segment)
                if 0 <= t and t <= 1 and u >= 0 and np.cross(r,s) != 0:
                    # intersection point with outer border, if it exists
                    intersection_point = p + t*r

                    # centers indicies
                    point_1 = points_index[0]
                    point_2 = points_index[1]

                    if la.norm(distances[points_index[0]]) < la.norm(intersection_point - vor.points[points_index[0]]):
                        distances[points_index[0]] = intersection_point - vor.points[points_index[0]]

                    if la.norm(distances[points_index[1]]) < la.norm(intersection_point - vor.points[points_index[1]]):
                         distances[points_index[1]] = intersection_point - vor.points[points_index[1]]


                    
            


        # return distance, and distance vector associated with voronoi verticies indicies
        return distances

    def fit(self,V,E,density=None):
        print("Fit")

        random.seed(self.random_state)

        # pick random points
        centroids = random.choices(list(V.keys()),k=self.n_cells)

        print(centroids)

        cell_pos = np.array([V[centroid] for centroid in centroids])

        print(cell_pos)

        # compute V.D of points
        vor = Voronoi(cell_pos)


        # this section will be pre-processing
        min_x = min(np.array(list(V.values()))[:,0])
        min_y = min(np.array(list(V.values()))[:,1])
        max_x = max(np.array(list(V.values()))[:,0])
        max_y = max(np.array(list(V.values()))[:,1])

        min_bound = [min_x,min_y]
        max_bound = [max_x,max_y]

        # I need to find the minimum, and maximum in the point set
        CH_points = np.array([min_bound,[max_bound[0],min_bound[1]],max_bound,[min_bound[0],max_bound[1]]])
        CH_segments = np.array([[CH_points[i-1],CH_points[i]] for i in range(len(CH_points))])



        # compute Objective Function

        O = self.objective_function(V,E,CH_points,CH_segments,vor)



        # loop

            # Calculate density of cells

            # calculate partial derivatives

            # calculate new cell positions

            # compute V.D of points

            # compare objective functions

        # return voronoi Cell Locations
        

        pass


class KMeans:
    def __init__(self,n_clusters=2, random_state=0,n_init=10,max_iter=300):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

    # compute k-means, and find clusters
    # V : Vertex Set -> Pos : [x,y]
    # E : Edge Set -> Weights : float
    def fit(self,V,E):
        # clamp two random verticies as centers
        centroids_iter = []
        measure_iter = []
        graph = adj_list_to_graph(E)

        # will return index of items
        verticies_pos = list(V.keys())
        neigh = NearestNeighbors(n_neighbors=1, radius=0.0000000000001) 
        neigh.fit(np.array(list(V.values())))


        # number of iterations
        for iteration in range(self.n_init):
            centroids = random.choices(list(V.keys()),k=self.n_clusters)
            measure_i = np.zeros(self.n_clusters)



            # computing clusters 
            for _ in range(self.max_iter):
                centroid_pos = np.zeros((self.n_clusters,2))
                centroid_count = np.zeros(self.n_clusters)
                measure_k = np.zeros(self.n_clusters)

                for v in list(V.keys()):
                    minimum_cost = float('inf')
                    minimum_c_index = 0

                    for index, centroid in zip(range(len(centroids)),centroids):
                        cost = find_path(graph,v,centroid).total_cost
                        if cost < minimum_cost:
                            minimum_cost = cost
                            minimum_c_index = index


                    centroid_pos[minimum_c_index] = centroid_pos[minimum_c_index] + V[v]
                    centroid_count[minimum_c_index] = centroid_count[minimum_c_index] + 1
                    measure_k[minimum_c_index] = measure_k[minimum_c_index] + minimum_cost

                centroid_count = np.array([1 if a == 0 else a for a in centroid_count])
                centroid_pos = centroid_pos/centroid_count[:,np.newaxis]
                Y = neigh.kneighbors(centroid_pos,1,return_distance=False)

                centroids = [verticies_pos[i[0]] for i in Y]
                
                # if there is next to no change from previous iteration
                if la.norm(measure_i-measure_k) < 1.0:
                    measure_i = measure_k
                    break
                measure_i = measure_k
                
            measure_iter.append(measure_i)
            centroids_iter.append(centroids)

        return centroids_iter[np.argmin(np.sum(measure_iter,axis=1))]



        

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
