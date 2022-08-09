import numpy as np
from dataclasses import dataclass
import astropy.units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from datetime import datetime
from numba import njit
from time import time


@njit
def calculate_distance_matrix(quick_list, 
                              angle_weight, 
                              time_weight, 
                              max_distance, 
                              min_ang_distance,
                              max_ang_distance):
    l = len(quick_list)
    distances = np.full((l,l), 2*max_distance)
    
    partitions = [0]
    for i in range(1,l):
        if (quick_list[i,2]-quick_list[i-1,2]) * time_weight > max_distance:
            partitions.append(i)
    partitions.append(l)
    
    for i in range(len(partitions)-1):
        for j in range(partitions[i], partitions[i+1]):

            for k in range(j+1, partitions[i+1]):
                distances[j,k] = distances[k,j] = calculate_distance(
                    quick_list[j],
                    quick_list[k],
                    angle_weight,
                    time_weight,
                    min_ang_distance,
                    max_ang_distance,
                    max_distance)
                
    np.fill_diagonal(distances,0.)
            
    return np.array(partitions), distances

@njit
def calculate_distance(point1, point2, angle_weight, time_weight, min_ang_distance, max_ang_distance, max_distance):
    ang_dis = np.arccos( np.clip(np.array([np.sin(point1[1]) * np.sin(point2[1]) 
                                           + np.cos(point1[1]) * np.cos(point2[1]) 
                                           * np.cos(point1[0] - point2[0])]),
                                 -1., 1.) )[0]
    
    if (ang_dis < min_ang_distance) or (ang_dis > max_ang_distance):
        return 2.*max_distance
    
    time_dis = abs(point1[2] - point2[2])
    
    return ( (angle_weight*ang_dis)**2 + (time_weight*time_dis)**2 )**0.5

@njit
def find_regions(distances, max_distance, partitions):
    regions = []
    
    for i,partition in enumerate(partitions[:-1]):
        unconnected = [j for j in range(partition, partitions[i+1])]
        
        while not len(unconnected)==0:
            temp_region = [unconnected.pop(0)]
            search_index = 0
            
            while search_index < len(temp_region):
                l = len(unconnected)
                for j in range(l-1,-1,-1):
                    if distances[ temp_region[search_index], unconnected[j] ] < max_distance:
                        temp_region.append(unconnected.pop(j))
                        
                search_index += 1
                
            regions.append(sorted(temp_region))
            
    return regions

def choose_random_weighted_interval(weights):
    r = np.random.random(1)[0] * np.sum(weights)
    s = 0.
    for i, w in enumerate(weights):
        s += w
        if r < s:
            return i
    return None
        
def calc_pair_combinations(number):
    return (number)*(number-1)/2
    
    
class Cluster:
    def __init__(self,
                 pointing: "Pointing",
                 query: "ClusteredQuery"):
        """
        Object that represents one cluster, and stores the Pointings within that cluster, as well as
        the average distance between pointings and the ClusteredQuery objected it was created within
        """
        
        self.indices = [pointing.index]
        self.avg_distance = 0.
        self.num_pointings = 1
        self.pointings = [pointing]
        self.query = query
        
    def add_pointing(self, pointing:"Pointing"):
        """Adds a new pointing to a cluster"""
        
        self.avg_distance = self.calc_new_avg_dist(pointing)
        self.indices.append(pointing.index)
        self.pointings.append(pointing)
        self.num_pointings += 1
        
    def should_add_pointing(self, pointing: "Pointing") -> bool:
        """Checks if a Pointing should be added to the Cluster"""
        if self.num_pointings < self.query._cluster_size_range[1]:
            if self.find_new_max_dist(pointing) < self.query._max_distance:
                if self.num_pointings >= self.query._cluster_size_range[0]:
                    if (self.calc_new_avg_dist(pointing)/self.avg_distance
                        < self.query._cluster_size_preference_threshold[self.num_pointings
                                                                        - self.query._cluster_size_range[0]]):
                        return True
                else:
                    return True
        return False
    
    def finalize_cluster(self):
        """Set the cluster attribute of the Pointings in the Cluster to the Cluster"""
        for p in self.pointings:
            p.cluster = self
            
    def calc_new_avg_dist(self, pointing: "Pointing") -> float:
        """Returns the new average distance between Pointings if a Pointing were added"""
        return ((self.avg_distance * calc_pair_combinations(self.num_pointings)
                 + np.sum(self.query._distances[pointing.index,self.indices]))
                 / calc_pair_combinations(self.num_pointings+1) )
        
    def find_new_max_dist(self, pointing: "Pointing") -> float:
        """Returns the maximum distance of a pointing with a cluster"""
        return np.amax(self.query._distances[pointing.index,self.indices])

    def find_closest_pointings(self, cluster2: "Cluster") -> tuple["Pointing"]:
        """Returns the Pointing pair with the smallest distance between two clusters"""
        d = self.query._distances[self.indices,:][:,cluster2.indices]
        c = np.unravel_index(d.argmin(), d.shape)
        return self.pointings[c[0]], cluster2.pointings[c[1]]
    

@dataclass
class Pointing:
    """
    Dataclass that represents a single Pointing. Contains its coordinates, scw_id, index,
    and its cluster may be added later
    """
    scw_id: str
    sky_coords: SkyCoord
    start_time: datetime
    index: int
    cluster: Cluster = None
        
    def angle_between_three_pointings(self, 
                                      pointing2: "Pointing", 
                                      pointing3: "Pointing", 
                                      angle_weight: float, 
                                      time_weight: float
                                      ) -> float:
        """Returns the angle between three different poitings
        Angle between the vector self -> poiting2 and vector self -> pointing3
        """
        
        origin = SkyCoord(self.sky_coords.ra.deg, (self.sky_coords.dec.deg%180.)-90., frame="icrs",unit="deg")
        origin_Frame = SkyOffsetFrame(origin = origin)
        
        p1 = self.sky_coords.transform_to(origin_Frame)
        p2 = pointing2.sky_coords.transform_to(origin_Frame)
        p3 = pointing3.sky_coords.transform_to(origin_Frame)
        
        a_a = p2.lon.deg - p3.lon.deg
        a_d2 = abs(p1.lat.deg - p2.lat.deg) * angle_weight
        a_d3 = abs(p1.lat.deg - p3.lat.deg) * angle_weight
        
        t_d2 = (pointing2.start_time - self.start_time).total_seconds()/86400 * time_weight
        t_d3 = (pointing3.start_time - self.start_time).total_seconds()/86400 * time_weight
        
        return np.arccos( np.clip((a_d2*a_d3*np.cos(np.deg2rad(a_a)) + t_d2*t_d3) 
                                  / np.linalg.norm([a_d2,t_d2]) / np.linalg.norm([a_d3,t_d3]), 
                                  -1., 1.) )
        

class ClusteredQuery:
    def __init__(self,
                 scw_ids: np.array, # has to be sorted by START_DATE
                 angle_weight: float,
                 time_weight: float,
                 max_distance: float,
                 min_ang_distance: float = 0.,
                 max_ang_distance: float = None,
                 cluster_size_range: tuple[int] = (5,5),
                 cluster_size_preference_threshold: tuple[float] = (),
                 failed_improvements_max: int = 3,
                 suboptimal_cluster_size: int = 2, # >= 1
                 close_suboptimal_cluster_size: int = 3, # >= suboptimal_cluster_size
                 track_performance: bool = False
                 ):
        """
        ClusteredQuery object. The main class that converts a list of scw_ids into Clusters
        composed of Pointings.
        
        :param scw_ids: numpy array with SCW_ID in first column, RA (deg) in second, DEC (deg) in third,
        and START_DATE (datetime) in the fourth column. Has to be sorted by start_date
        
        :param angle_weight: determines how much effective distance a seperation of 1 degree is
        
        :param time_weight: determines how much effective distance a seperation of 1 day is
        
        :param max_distance: the maximum effective distance two pointings in a cluster may have
        
        :param min_ang_distance: the minimum angular separation (deg) two pointing in a cluster may have
        
        :param max_ang_distance: the maximum angular separation (deg) two pointing in a cluster may have
        It is recommended to only specify this parameter if angle_weight=0.
        
        :param cluster_size_range: the smallest and largest cluster sizes the algorithm aims to create
        
        :param cluster_size_preference_threshold: the thresholds for average distance ratios that are considered
        when adding a Pointing to Cluster within the cluster_size_range
        Example: if cluster_size_range = (3,5), then cluster_size_preference_threshold = (4.,5.) would mean that
        a fourth pointing is only added to a cluster of 3 if the average distance does not increase four-fold,
        and a fith would only be added if the average distance does not increase five-fold
        Has to contain one element for each cluster size within the cluster_size_range
        
        :param failed_improvements_max: how many successive failed improvements are allowed before a 
        Region is declared finished
        Use this parameter to determine the trade-off between run time and quality of clusters
        
        :param suboptimal_cluster_size: the maximum cluster size that is used to begin an improvement
        Cannot be smaller than 1
        
        :param close_suboptimal_cluster_size: the maximum cluster size that is paired with a suboptimal cluster
        to attempt an improvement.
        Cannot be smaller than suboptimal_cluster_size
        
        :param track_performance: Should the algorithm print various performance statistics after running?
        """
        
        assert len(cluster_size_preference_threshold) == (cluster_size_range[1] - cluster_size_range[0]), \
                "cluster_size_preference_threshold must cointain 1 element for each cluster_size_range"
        
        assert suboptimal_cluster_size >= 1, "suboptimal_cluster_size must be at least 1"
        
        assert close_suboptimal_cluster_size >= suboptimal_cluster_size, \
                "close_suboptimal_cluster_size must be at least suboptimal_cluster_size"
        
        self._angle_weight = float(angle_weight)
        self._time_weight = float(time_weight)
        self._max_distance = float(max_distance)
        self._min_ang_distance = float(min_ang_distance)
        self._cluster_size_range = cluster_size_range
        self._cluster_size_preference_threshold = cluster_size_preference_threshold
        self._failed_improvements_max = failed_improvements_max
        self._suboptimal_cluster_size_range = (1, suboptimal_cluster_size)
        self._close_suboptimal_cluster_size_range = (1, close_suboptimal_cluster_size)
        
        if max_ang_distance is None:
            self._max_ang_distance = 360.
        else:
            self._max_ang_distance = float(max_ang_distance)
            
        assert self._max_ang_distance > self._min_ang_distance, "max_ang_distance must > min_ang_distance"
            
        if self._angle_weight == 0.:
            self._angle_angle_weight = self._max_distance / self._max_ang_distance
        else:
            self._angle_angle_weight = self._angle_weight
                    
        self._num_pointings = len(scw_ids)
        
        self._track_performance = track_performance
        if self._track_performance:
            start_time = time()
            self._region_sizes = {}
            self._initial_cluster_sizes = self.initialize_size_dictionary()
            self._cluster_sizes = self.initialize_size_dictionary()
            for i in self._initial_cluster_sizes.keys():
                self._initial_cluster_sizes[i] = 0
                self._cluster_sizes[i] = 0
            self._attempted_improvements = 0
            self._implemented_improvements = 0
            self._dead_ends = 0
        
        
        quick_list = np.zeros((self._num_pointings, 3))
        quick_list[:,0:2] = scw_ids[:,1:3]
        quick_list[:,0:2] = np.deg2rad(quick_list[:,0:2])
        for i in range(self._num_pointings):
            quick_list[i,2] = (scw_ids[i,3] - datetime(2000,1,1,0,0,0)).total_seconds()/86400  
            
        partitions, self._distances = calculate_distance_matrix(
            quick_list, np.rad2deg(self._angle_weight), self._time_weight, self._max_distance, 
            np.deg2rad(self._min_ang_distance), np.deg2rad(self._max_ang_distance)
            )
                
        self._region_indices = find_regions(self._distances, self._max_distance, partitions)
        
        self._pointings = np.array(
            [Pointing(pointing[0], 
                      SkyCoord(pointing[1],pointing[2],frame="icrs",unit="deg"),
                      pointing[3], 
                      index)
             for index, pointing in enumerate(scw_ids)]
            )
        
        self.clusters = self.initialize_size_dictionary()
        
        for i in self._region_indices:
            Region(i, self)
            
        if self._track_performance:
            for i in self._region_indices:
                if len(i) in self._region_sizes:
                    self._region_sizes[len(i)] += 1
                else:
                    self._region_sizes[len(i)] = 1
            for clusters in self.clusters.values():
                for cluster in clusters:
                    self._cluster_sizes[cluster.num_pointings] += 1
            
        # print()
        # print("All Done")
        # clustered = set()
        # n = 0
        
        # for key, value in self.clusters.items():
        #     for c in value:
        #         print(f"{c.indices}, {c.avg_distance}")
        #         clustered = clustered | set(c.indices)
        #         n += c.num_pointings
                
        # missing = [i for i in range(self._num_pointings) if i not in clustered]
        # print()
        # print(missing)
        # print(self._num_pointings, n)
        # print()
        
        if self._track_performance:
            end_time = time()
            print("Performance Review!")
            print()
            print("Run Time:")
            print(f"{(end_time - start_time):.2f}s")
            print()
            print("Region Sizes:")
            print(sorted(self._region_sizes.items()))
            print()
            print("Initial Cluster Sizes:")
            print(self._initial_cluster_sizes)
            print()
            print("Final Cluster Sizes:")
            print(self._cluster_sizes)
            print()
            print("Attempted Improvements:")
            print(self._attempted_improvements)
            print()
            print("Implemented Improvements:")
            print(self._implemented_improvements)
            print()
            print("Dead-End Clustering Paths:")
            print(self._dead_ends)
            
                    
    def initialize_size_dictionary(self) -> dict[list]:
        """Returns dictionary with empty list for each cluster size"""
        dict = {}
        for i in range(self._cluster_size_range[1]):
            dict[i+1]=[]
        return dict
    
    def get_clustered_scw_ids(self) -> dict[list[str]]:
        """Returns dictionary with lists of SCW_IDs in clusters, arranged by cluster size"""
        scw_id_clusters = self.initialize_size_dictionary()
        for size, clusters in self.clusters.items():
            for cluster in clusters:
                scw_id_clusters[size].append([p.scw_id for p in cluster.pointings])
        return scw_id_clusters

    
class Region:
    def __init__(self,
                 region_indices: list,
                 query: ClusteredQuery
                 ):
        """
        Object created by ClusteredQuery that clusters pointings together
        :param region_indices: list of indices that are to be clustered by this Region
        :param query: the ClusteredQuery object that created this Region
        """
        
        self.indices = region_indices
        self.query = query
        
        self.clusters = self.query.initialize_size_dictionary()
        self.potential_clusters1 = self.query.initialize_size_dictionary()
        self.potential_clusters2 = self.query.initialize_size_dictionary()
            
        # print()
        # print("Initial Clustering!")
        # print()
        self.initial_clustering()
        
        if self.query._track_performance:
            for clusters in self.clusters.values():
                for cluster in clusters:
                    self.query._initial_cluster_sizes[cluster.num_pointings] += 1
        
        # for key, value in self.clusters.items():
        #     for c in value:
        #         print(f"{c.indices}, {c.avg_distance}")
        
        
        failed_improvements = 0
        while failed_improvements < self.query._failed_improvements_max and self.has_suboptimal_clusters():
            if self.query._track_performance:
                self.query._attempted_improvements += 1
                
            if not self.attempt_improvement():
                failed_improvements += 1
                
            else:
                failed_improvements = 0
                if self.query._track_performance:
                    self.query._implemented_improvements += 1
                
        for size, clusters in self.clusters.items():
            self.query.clusters[size].extend(clusters)
        
        # print()
        # print()
        # print("Finishing Region!")
        # print()
        # print()
                
    
    def initial_clustering(self):
        """Function that clusters previously unclustered pointings"""
        for position, index in enumerate(self.indices):
            if self.query._pointings[index].cluster is None:
                # print()
                # print(f"New Cluster: {position}, {index}")
                cluster = Cluster(self.query._pointings[index], self.query)
                self.add_following_pointings(cluster, position+1)
                cluster.finalize_cluster()
                self.clusters[cluster.num_pointings].append(cluster)
                
        
    def add_following_pointings(self, cluster: Cluster, position: int):
        """Called by initial_clustering, tries to add succesive Pointings to Cluster"""
        if cluster.num_pointings < self.query._cluster_size_range[1]:
            
            # Performance may be optimized via this value
            for position2, index in enumerate(
                self.indices[position : position + 20*self.query._cluster_size_range[1]]
                ):
                
                if self.query._pointings[index].cluster is None:
                    # print(f"Checking: {position+position2}, {index}")
                    if cluster.should_add_pointing(self.query._pointings[index]):
                        # print(f"Adding: {position+position2}, {index}")
                        cluster.add_pointing(self.query._pointings[index])
                        self.add_following_pointings(cluster, position+position2+1)
                        break
        
    def attempt_improvement(self) -> bool:
        """Function that attempts to improve a clusering
        It does this by connecting a suboptimal_cluster to a close_suboptimal_cluster
        via a path of connecting clusters, then reclusters all of the selected clusters,
        and finally checks if the resulting clustering is an improvement over the existing clustering.
        If yes, it will keep the new clustering, if no then not
        """
        # print()
        # print("Attempting Improvement!")
        # print()
        c1 = self.find_suboptimal_cluster()
        c2 = self.find_close_suboptimal_cluster(c1)
        found_path, recluster_indices = self.find_cluster_path(c1,c2)
        if not found_path:
            return False
        else:
            self.recluster_pointings(recluster_indices, c1)
        
        # print()
        # print("Done!")
        # print()
        # for key, value in self.clusters.items():
        #     for c in value:
        #         print(f"{c.indices}, {c.avg_distance}")
        # print()
        # for key, value in self.potential_clusters1.items():
        #     for c in value:
        #         print(f"{c.indices}, {c.avg_distance}")
        # print()
        # for key, value in self.potential_clusters2.items():
        #     for c in value:
        #         print(f"{c.indices}, {c.avg_distance}")
        # print(f"Comparison: {self.calc_clustering_cost(self.potential_clusters1)}, {self.calc_clustering_cost(self.potential_clusters2)}")
        
        if self.calc_clustering_cost(self.potential_clusters1) > self.calc_clustering_cost(self.potential_clusters2):
            self.implement(self.potential_clusters2, True)
            # print()
            # print("Implemented!")
            # for key, value in self.clusters.items():
            #     for c in value:
            #         print(f"{c.indices}, {c.avg_distance}")
            # print()
            return True
        else:
            self.implement(self.potential_clusters1, False)
            # print()
            # print("Rejected!")
            # for key, value in self.clusters.items():
            #     for c in value:
            #         print(f"{c.indices}, {c.avg_distance}")
            # print()
            return False
        
    
    def find_suboptimal_cluster(self) -> Cluster:
        """Chooses a cluster in the suboptimal_cluser_size_range"""
        size_weights = np.array([len(self.clusters[i]) / i**2
                                 for i in range(self.query._suboptimal_cluster_size_range[0],
                                                self.query._suboptimal_cluster_size_range[1] + 1)])
        
        size = choose_random_weighted_interval(size_weights) + 1
        index = np.random.randint(len(self.clusters[size]))
        
        cluster = self.clusters[size].pop(index)
        self.potential_clusters1[size].append(cluster)
        
        return cluster
    
    def find_close_suboptimal_cluster(self, cluster: Cluster) -> Cluster:
        """Chooses a close cluster in the close_suboptimal_cluster_size_range"""
        clusters = []
        cluster_size_indices = [0]
        
        for i in range(self.query._close_suboptimal_cluster_size_range[0], 
                       self.query._close_suboptimal_cluster_size_range[1] + 1):
            clusters.extend(self.clusters[i])
            cluster_size_indices.append( len(self.clusters[i]) + cluster_size_indices[i-1] )
            
        cluster_weights = np.zeros(len(clusters))
        
        # Performance may be optimized via this value
        for i, c in enumerate(clusters):
            d = self.query._distances[cluster.indices,:][:,c.indices]
            cluster_weights[i] = np.exp( -5.*np.amin(d) / self.query._max_distance )
            
        for i in range(1, self.query._close_suboptimal_cluster_size_range[1] + 1):
            cluster_weights[cluster_size_indices[i-1]:cluster_size_indices[i]] /= i
        
        index = choose_random_weighted_interval( cluster_weights )
        
        for size, csi in enumerate(cluster_size_indices):
            if not index >= csi:
                break
        
        true_index = index - cluster_size_indices[size-1]
        
        cluster2 = self.clusters[size].pop(true_index)
        self.potential_clusters1[size].append(cluster2)
        
        return cluster2
        
    def find_cluster_path(self, cluster1: Cluster, cluster2: Cluster) -> tuple[bool, set]:
        """Connects two clusters via a path of inbetween clusters"""
        indices_in = set(cluster1.indices)
        indices_to = set(cluster2.indices)
        arrived = False
        
        while not arrived:
            indices_out = np.array([i for i in self.indices if i not in indices_in])
            pointing1, pointing2 = cluster1.find_closest_pointings(cluster2)
            # print(pointing1.index, pointing2.index, self.query._distances[pointing1.index, pointing2.index])

            # Performance may be optimized via this value
            close_indices = self.find_closest_points(pointing1.index, indices_out, 5)
            # print(close_indices)
            
            angle = np.vectorize(lambda p: pointing1.angle_between_three_pointings(
                pointing2, p, self.query._angle_angle_weight, self.query._time_weight
                ))
            
            angles_filtered = angle(self.query._pointings[close_indices])
            
            distances_filtered = self.query._distances[pointing1.index, close_indices]
            
            # Performance may be optimized via this value
            close_indices_weights = ((distances_filtered < self.query._max_distance) 
                                     * np.exp(-4. * distances_filtered / self.query._max_distance) 
                                     * np.cos(angles_filtered/2.)**8)
            
            # print(angles_filtered)
            # print(distances_filtered)
            # print(close_indices_weights)
            
            random_index = choose_random_weighted_interval(close_indices_weights)
            
            if (random_index is None) or (np.amax(np.cos(angles_filtered)) < 0.):
                # print("Cannot go forwards, or reached dead end!")
                self.implement(self.potential_clusters1, False)
                if self.query._track_performance:
                    self.query._dead_ends += 1
                return False, None

            index = close_indices[random_index]
            
            cluster1 = self.query._pointings[index].cluster
            indices_in = indices_in | set(cluster1.indices)
            if index in indices_to:
                arrived = True
            else:
                self.clusters[cluster1.num_pointings].remove(cluster1)
                self.potential_clusters1[cluster1.num_pointings].append(cluster1)
        
        return True, indices_in
        
        
    def recluster_pointings(self, recluster_indices: list[int], start_cluster: Cluster):
        """Takes existing clusters and reclusters them into new clusters"""
        # Performance may be optimized via this value
        index = start_cluster.indices[
            choose_random_weighted_interval(np.exp(
                np.average(self.query._distances[start_cluster.indices,:][:,list(recluster_indices)], axis=1) 
                * 4. / self.query._max_distance))
            ]
        
        cluster = Cluster(self.query._pointings[index], self.query)
        already_clustered = {index}
        not_clustered = np.array([i for i in recluster_indices if i not in already_clustered])
        
        # print()
        # print("Recluster Pointings!")
        
        added_all_clusters = False
        while not_clustered.size != 0:
            # print()
            # print("New Cluster!")

            cluster_done = False
            
            while not cluster_done:
                # print()
                # print("Continue Cluster!")
                # print(index, cluster.indices)
                # print(already_clustered)
                # print(not_clustered)
                
                # Performance may be optimized via this value
                close_indices = self.find_closest_points(index, not_clustered, 5)
                
                internal_distances = np.average(self.query._distances[cluster.indices,:][:,close_indices], axis=0)
                external_distances = np.average(self.query._distances[close_indices,:][:,close_indices], axis=0)
                
                # Performance may be optimized via this value
                close_indices_weights = np.exp( (5.*external_distances - 4.*internal_distances) 
                                                / self.query._max_distance )
                
                # print(close_indices)
                # print(internal_distances)
                # print(external_distances)
                # print(close_indices_weights)
                
                added = False
                while not added:
                    random_index = choose_random_weighted_interval(close_indices_weights)
                    # print(random_index)
                    if random_index is None:
                        cluster_done = True
                        # print("Found None!")
                        break
                    
                    index = close_indices[random_index]
                    # print(index)
                    if cluster.should_add_pointing(self.query._pointings[index]):
                        added = True
                        cluster.add_pointing(self.query._pointings[index])
                        already_clustered.add(index)
                        not_clustered = np.array([i for i in recluster_indices if i not in already_clustered])
                        
                        # print("Added!")
                        
                        if cluster.num_pointings == self.query._cluster_size_range[1]:
                            cluster_done = True
                        
                    else:
                        close_indices_weights[random_index] = 0.
                
                if not_clustered.size == 0:
                    self.potential_clusters2[cluster.num_pointings].append(cluster)
                    added_all_clusters = True
                    break
                
                if cluster_done:
                    self.potential_clusters2[cluster.num_pointings].append(cluster)
                    
                    # Performance may be optimized via this value
                    weights = np.exp(4.*external_distances / self.query._max_distance)
                     
                    if added:
                        weights[random_index] = 0.
                    index = close_indices[choose_random_weighted_interval(weights)]
                    
                    cluster = Cluster(self.query._pointings[index], self.query)
                    already_clustered.add(index)
                    not_clustered = np.array([i for i in recluster_indices if i not in already_clustered])
                    break
                    
                else:
                    index = cluster.indices[np.random.randint(cluster.num_pointings)]
                    
        if not added_all_clusters:
            self.potential_clusters2[cluster.num_pointings].append(cluster)
            
    def find_closest_points(self, start_index: int, search_indices: list[int], number_multiplier: int) -> list[int]:
        """Returns a list of the pointings closest to a pointing froma list of pointings"""
        num_points = min(len(search_indices), self.query._cluster_size_range[1]*number_multiplier)
        sortable_pointings = [i for i in range(num_points)]
        distances = self.query._distances[start_index,search_indices]
        return search_indices[np.argpartition(distances, sortable_pointings)[:num_points]]
            
    def calc_clustering_cost(self, cluster_dict: dict[list[Cluster]]) -> float:
        """Return a cost for a clustering configuration. Lower is better"""
        n = 0
        avg_d = 0.
        for clusters in cluster_dict.values():
            for cluster in clusters:
                n += 1
                if cluster.num_pointings == 1:
                    avg_d += self.query._max_distance
                else:
                    avg_d += cluster.avg_distance

        return n + avg_d/n / self.query._max_distance
    
    def has_suboptimal_clusters(self) -> bool:
        """Checks if enough suboptimal clusters exist to attempt an improvment"""
        s = 0
        for i in range(self.query._suboptimal_cluster_size_range[0], 
                       self.query._suboptimal_cluster_size_range[1] + 1):
            s += len(self.clusters[i])
        if s >= 2:
            return True
        elif s >= 1:
            for i in range(self.query._suboptimal_cluster_size_range[1] + 1,
                           self.query._close_suboptimal_cluster_size_range[1] + 1):
                s += len(self.clusters[i])
            if s >= 2:
                return True
        return False
    
    def implement(self, potential_clusters: dict[list[Cluster]], finalize: bool):
        """Implements a clustering configuration by adding it to the main cluster dictionary"""
        for size, clusters in potential_clusters.items():
            if finalize:
                for cluster in clusters:
                    cluster.finalize_cluster()
            self.clusters[size].extend(clusters)
        self.potential_clusters1 = self.query.initialize_size_dictionary()
        self.potential_clusters2 = self.query.initialize_size_dictionary()
