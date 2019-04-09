Ground Truth Transition Matching
Kelvin Lin (kelvin.lin1@gmail.com
AI City 2019

Overview
	This code is intended to find the vehicle transitions in the single-camera tracking ground truth and to
	identify the transition by the cameras, waypoints, and vectors. 

	The results of the program are stored in
	result.txt which is generated in every code execution. The program will also generate two temporary files: 
	output.log and sct_gt_verified.txt; the first file is used for debugging and logging values and the second
	file is used to speed up the runtime by saving a intermediate file (the filtered ground truth).

Filesystem
	/
		cam_timestamp/
		ref/
		0319camera_link/
		main.py
		functions.py

Descriptions
	cam_timestamp/
		This is the collection of text files that define the each camera's time offset in seconds.

	ref/
		This folder contains all the relevant reference files. See the top of main.py for each file's 
		description.

	0319camera_link/
		This folder contains all the other reference files that are not directly used in main.py

	main.py
		This is the top-level script that loads all the data in the reference files into Python
		data structures and then also calls the routines to perform the matching. 

		This file has one parameter (verify) that is used to signify whether the program should
		filter out vehicles that appear in only one camera. 

	functions.py
		This contains all the implementation details of how the ground truth is processed and how the 
		matching is done. See Implementation for specific function descriptions.

		This file has 5 parameters:
			framerate specifies the fps of the videos. It is used for time stamp normalization.
			
			trajectory_frames specifies how many frames should be sampled to determine the vehicle trajectory.

			zone_width specified the width/height of the bounding region centered about the vector midpoint is. This
			is currently used in the IOU eval.

			IOU_weight and cos_weight are the weights of the IOU and cosine similarity in the evaluation function. 
			The evaluation function is used to determine the vectors from the trajectory.

		Requires numpy, math, scipy, itertools, tqdm packages

Implementation
	The general flow of the matching process is as follows:
		1. Preprocess ground truth by filtering out all vehicles that do not meet multi-camera criteria
			i.e. vehicles that only appear in one camera in the ground truth
		2. Create a list of all transition pairs. Transition pairs are identified by the camera and the trajectory.
		3. Using the transition pair information, identify the vectors that best match the transition. Using lookup 
		tables, determine the waypoint and all other desired information.
		4. Generate a list of all relevant values for the transition ground truth and write to output file.

	In the following sections, the implementation details are described for each step in the general flow.
	The code is also thoroughly commented, so please also reference those comments to see what each function does.

	Step 1. Ground Truth Preprocessing
		Since we are only interested in transitions between cameras, all vehicles that do not cross cameras do not need 
		to be considered. In this initial step, vehicle ids that fall under this classification are removed and an 
		intermediate ground truth file (sct_gt_verified.txt) is generated. This step only runs when the parameter 
		verify = TRUE.

		This step is implemented in functions.py:filter_single_camera_ids. The only argument to this function is sct_gt
		which is the single-camera tracking ground truth. The function identifies all vehicle ids that do not appear in 
		more than 1 camera and removes all ground truth entries for that vehicle.

		First, the function identifies all unique vehicle ids in the ground truth by reading the 2nd element of each row
		of the ground truth data structure. For each vehicle id number, all the ground truth entries that have the same vehicle id are extracted.
		Within all the ground truth entries, all the unique camera ids are extracted. If the number of unique camera values is less
		than 2, then the vehicle is said to have only appear in a single camera. In this case, all the ground truth entries
		for that vehicle are removed from the ground truth data structure.

	Step 2. Create a list of all transition pairs
		In this step, all the transition pairs are identified in the ground truth and a list of transition pairs is generated. A 
		transition pair in this case is defined by two camera ids, the frame number, and the vehicle location, trajectory and bounding
		box. This step is implemented in functions.py:create_transition_pairs and functions.py:get_vehicle_trajectory.

		The create_transition_pairs function is the main routine for this step. First, all the unique vehicle ids are found.
		For each vehicle, all ground truth entries that match the vehicle id are retrieved (called vehicle ground truth). The vehicle
		ground truth is parsed to determine the frame range that the vehicle appear in each camera. During this step, the frame offsets are 
		applied. These offsets (originally in seconds) are converted into frames for ease of comparison. 
		NOTE: this might cause issues when reading the frame numbers! Feel free to change this implementation to do the comparison in seconds.

		For each camera the frame ranges are described as a two-element list [first frame, last frame]. The frame ranges are generated by stepping 
		through the entire vehicle ground truth and doing a frame compare with a current entry in the frame range dictionary. In this dictionary, 
		the keys are the camera id and the values are the frame range. After all the frame ranges for each camera is identified for the vehicle, we 
		identify the "nearest neighbor" cameras. A nearest neighbor camera is another camera whose frame range is DISJOINT with the current camera,
		but also whose last frame is the closest to the current camera's start frame out of all the cameras.
		
		For example, if we have cameras A, B, C with the following frame ranges:
			A = [5, 10]
			B = [1, 2]
			C = [3, 4]
		In this case, C's nearest neighbor is B and A's nearest neighbor is C. B has no nearest neighbor because it has no camera that occur before it.
		Note: if another camera is not disjoint with the current camera, but also is the closest to the current camera, then it is not the nearest neighbor
		but no other cameras can be the nearest neighbor either.
		If we change the above example to:
			A = [5, 10]
			B = [3, 6]
			C = [1, 2]
		C is not the nearest neighbor to A, because B is closer.

		After all the nearest neighbors have been identified for each camera, each possible camera pair is evaluated. This can become extremely expensive
		as the camera pairs increase as it is a N choose 2 possibilities, where N is the number of cameras that a vehicle is seen by. For each possible pair,
		we evaluate whether or not a transition existing by the frame ranges. 
		There are three cases where a transition can exist. We define source and destination to be the camera pair where a vehicle travels from source to destination.
			1. The first frame of the destination camera occurs in the source camera's frame range
			    src 	.......O............
            	dest           O..............
            2. The last of the destination camera occurs in the source camera's frame range
			    src 	    .........O.......
            	dest    .............O
            3. Source camera is the nearest neighbor to destination
            	src 	......O
            	dest             O............
            	other	...|						This is not the nearest neighbor because src is closer

        Based on this criteria, the pair is identified as a transition or not a transition. If it is not a transition, then the program will move onto the next pair.
        If it is a transition, then the trajectory is identified by calling the get_vehicle_trajectory function. 

        Based on the case that the transition was classified as, we determine a direction to sample the frames for the vehicle trajectory. Exits are sampled backwards
        from the transition frames; Entries are sampled forward from the transition frames. The trajectory is determined by taking the difference of the center of the bounding box
        for the vehicle over the sampled frames. The location is the averaged center of the bounding box over the sampled frames. The transition bounding box is
        the list of bounding boxes used to generate the result.

        For both the source and destination cameras, the camera id, transition frame, trajectory, location and bounding boxes are saved.
        All these values are compiled into a transition record and inserted into the transition dictionary. Each vehicle has a dictionary entry of all its 
        transitions in this structure.

	Step 3. Identifying vectors
		This step is implemented in the functions.py:identify_vectors_waypoints, functions.py:match_vector and functions.py:bb_interserction_over_union routines.
		In this step, for every vehicle id, each of the transition records are processed and the following values are determined:
			Output format (space delimited)
	                index   value
	                0       camA_ID 
	                1       camB_ID
	                2       frame_difference 
	                3       waypointA_ID-vector_type 
	                4       waypointB_ID-vector_type 
	                5-6     waypointA_start_point
	                7-8     waypointB_start_point
	                9-10    waypointA_end_point 
	                11-12   waypointB_end_point 
	                13-15   waypointA_GPS 
	                16-18   waypointB_GPS 
	                19-20   waypointA_vector 
	                21-22   waypointB_vector
	                23-24   vectorA_bbox
	                25-26   vectorB_bbox
	                27-28   trajectoryA_bbox
	                29-30   trajectoryB_bbox

	    The vectors are identified in the match_vector function. In this function, an evaluation function utilizing the IOU and cosine similarity of the
	    vehicle and vector bounding regions and trajectories, respectively, is used. Using the camera id from the transition record, all the vectors that 
	    are associated with that camera are retrieved. Then, for each of the vectors, the IOU and cosine similarity is computedand then the evaluation function 
	    is applied. The vector that has highest evaluation is determined to be the match. 

	    After the vector has been matched, the waypoint is identified by looking up the matched vector id in the table. All the relevant vector information is
	    also searched using the vector lookup table.

	    FUTURE TODO: Implement Zone-based matching instead of vector matching.
	    To implement zone-based matching, the following is needed:
	    	camera to zone table
	    	waypoint to zone table (if you need the waypoints)

	    Modifications needed
	    	match_vector implemented the evaluation function for finding the matching vectors. You need to pass in the camera to zone table and perform the
	    	IOU using each zone bounding box and the trajectory bounding box. Depending on what information you need from this function, you can change the 
	    	return values. 



	Step 4. Writing to output file
		In this step, the output from the vector identification is written to a file using the write_file function.

