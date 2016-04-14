'''
    This code is intended to be run in the IPython shell. 
    You can enter each line in the shell and see the result immediately.
    The expected output in the Python console is presented as commented lines following the
    relevant code.
'''

%pylab inline
# Populating the interactive namespace from numpy and matplotlib

path = "/PATH/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
ae = imread(path)
imshow(ae)

tmpPath = "/tmp/aeGray.jpg"
aeGary = imread(tmpPath)
imshow(aeGary, cmap=plt.cm.gray)

pc = np.loadtxt("/tmp/pc.csv", delimiter=",")
print(pc.shape)
# (2500, 10)
def plot_gallery(images, h, w, n_row=2, n_col=5):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[:, i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title("Eigenface %d" % (i + 1), size=12)
        plt.xticks(())
        plt.yticks(())

plot_gallery(pc, 50, 50)

s = np.loadtxt("/tmp/s.csv", delimiter=",")
print(s.shape)
plot(s)
# (300,)
plot(cumsum(s))
plt.yscale('log')