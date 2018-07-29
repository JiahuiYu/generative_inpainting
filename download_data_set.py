import errno
import os
import urllib.request


##todo: delete this shit

def download_img(set_name="doors", num=0, url="http://textures.forrest.cz/textures/library/2009_doors/1.jpg"):
    path_dir = "data_sets_our/" + set_name + "/"
    check_path_or_open(path_dir)
    with urllib.request.urlopen(url) as img_url:
        with open(path_dir + str(num) + ".jpg", 'wb') as f:
            f.write(img_url.read())


def check_path_or_open(path_dir):
    if not os.path.exists(os.path.dirname(path_dir)):
        try:
            os.makedirs(os.path.dirname(path_dir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def download_images(set_name="doors", url_prefix="http://textures.forrest.cz/textures/library/2009_doors/"):
    i = 1
    while True:
        u = url_prefix + str(i) + ".jpg"
        try:
            print("downloading from: " + u)
            download_img(set_name, num=i, url=u)
        except:
            print("exception occured")
            return
        i += 1


url_suffixes = [
    "http://textures.forrest.cz/textures/library/2009_doors/",
    "http://textures.forrest.cz/textures/library/bump/Blast",
]

for url_suf in url_suffixes:
    download_images("blast", url_suf)
