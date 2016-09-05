import gzip
import sys

def validate(filename):  

  try:
    keys = []

    with gzip.open(filename) as f:
      for i, line in enumerate(f):
        this_line = line.split()
        if len(this_line) != 51:
          return False, "Wrong number of fields : line " + str(i)
        try:
          float_line = [float(x) for x in this_line[1:]]
        except ValueError:
          return False, "Unable to convert to float : line " + str(i)

        keys.append(this_line[0])

    keys = [key.lower() for key in keys]
    if not len(keys) == len(set(keys)):
      return False, "Multiple vectors present for some key"

    return True, ""
  except IOError as e:
    return False, e.message
 

def main():
  vector_file = sys.argv[1]
  result, message = validate(vector_file)
  if result:
    print "Success, the file " + vector_file + " is ready to upload"
  else:
    print "The file " + vector_file + " is not ready for upload: "+message

if __name__ == "__main__":
  main()
