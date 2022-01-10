class TrackableObject:
	def __init__(self, objectID, centroid):
        # sotocker le ID object, puis initailiser les liste de centres 
		self.objectID = objectID
		self.centroids = [centroid]

        # initialiser boolean pour indiquer si l'object qui'est détecter par notre systéme si compter ou no.
		self.counted = False
