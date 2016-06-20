import h5py

filename= "checkpoint.02--0.26--0.93.hdf5"

def printname(name):
	print name

with h5py.File(filename, "r", driver=None) as f:
	f.visit(printname)
	for k in f.keys():
		print "**: ", f[k].keys()
		val = f[k]
		for k in val.keys():
			try:
				vv = val[k].shape
				print "***: ", vv
			except:
				pass
	quit()


with h5py.File(filename, "r", driver=None) as f:
	print help(h5py.File)
	print h5py.Group.visit("ge_dense1/ge_dense1_W")
	quit()
	#print f.keys()
	print f["dense_1"].keys()
	ff = f["dense_1"]
	print ff["dense_1_W"]
	w = ff["dense_1_W"]
	print "w[1,1]= ", w[:,:]
	print ff["dense_1_b"]

	bb = ff["dense_1_b"]
	print "bb[1,1]= ", bb[:]
	print bb

quit()

print help(hdf5)
print dir(hdf5)
#checkpoint.06.hdf5

#----------------------------------------------------------------------
dense_1/dense_1_W
dense_1/dense_1_b
dense_2
dense_2/dense_2_W
dense_2/dense_2_b
dropout_1
ge_dense1
ge_dense1/ge_dense1_W
ge_dense1/ge_dense1_b
ge_dropout1

