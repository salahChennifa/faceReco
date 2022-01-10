# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50):
		# permet de compter par donner un IDs pour chaque object
		self.nextObjectID = 0
		# c'est un dictionnaire qui utilise objectID comme clé et centre cord(x,y) comme valeur
		self.objects = OrderedDict()
		# Dictinnaire qui stocker nombre de frame consécutives comme valeur et ID de l'object qui a été marqué comme 'perdu'
		self.disappeared = OrderedDict()

		# stocker le nombre de max cosécutives frames pour marquer un object comme 'perdu' pour déregister
		self.maxDisappeared = maxDisappeared

		# stocker la distance max entre la centre et a neauveau object ---si la distance est
		# sup à max donc est marquer 'perdu'
		self.maxDistance = maxDistance

	def register(self, centroid):
		# prend comme paramétre a centroid et ajouter ce centre à  dict objects
		self.objects[self.nextObjectID] = centroid
		# le nombre de fois est initialiser à 0
		self.disappeared[self.nextObjectID] = 0
		# incrémenter pour un nouvelle object ID
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# pour déregister notre tracker de object et les perdus
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# if la list de rectangles est de longuer 0
		if len(rects) == 0:

			# boucler sur les existance object suivi et marquer 'perdu'
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				# si on a rechercher sur les frames un object et rest perdu et sup a maxDisappared
				# donc fait un deregister
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			# retourne les neauveau
			return self.objects
		# on d'autre cas on a
		# initialisation d'un array pour stocker les centre des rectangles pour ce frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")


		# boucler sur les ractangles qui est exister dans le frame
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# utilizer le coordonnee de chaque rectangle pour calculer la centre.
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# si on a aucun object à suivi
		# on fait une registration pour inputCentoid
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])


		#  d'autres cas on a besion de mise à jour pour les existance objects
		# basant sur la distance euclidienne.
		else:
			# différnce notre ID à les différentes centre:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# calcule la distance entre chaque pair de centre existe et
			# de neveau centre aparaitre
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			# toruver min value dans chaque row et fait une ordre petit à grand
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			# meme chose trouver le min dans chaque colonne et ordorer ensuite
			# notre but est d'avoir valeur indexe avec plus petit distance:
			cols = D.argmin(axis=1)[rows]


			# afin de déterminer si nous devons mettre à jour, enregistrer ou désenregistrer un#
			# objet, nous devons
			# garder une trace des index des lignes et des colonnes que nous avons déjà examinés
			usedRows = set()
			usedCols = set()

			# boucler sur conbination de (row, column) index
			for (row, col) in zip(rows, cols):
				# si nous avons examannée déja donc ignorer
				if row in usedRows or col in usedCols:
					continue

				# if the distance between centroids is greater than
				# the maximum distance, do not associate the two
				# centroids to the same object
				# si la distance entre les centres est sup à la max distance,
				# jamais associer pour chaque object deux centre
				if D[row, col] > self.maxDistance:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				# d'autre cas, prendre actual ID object  et crée un
				# neaveau centre et reset disparé compteur à 0
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects