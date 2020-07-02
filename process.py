import os
import numpy as np

if __name__ == '__main__':
    import argparse

    # Current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print('Working from ' + current_dir)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Split dataset for cross-validation')
    parser.add_argument('--dataset', nargs='?', const=1, metavar='/path/to/dataset',
                        default=os.path.join(current_dir, 'Images/leaks'),
                        help='Directory of the dataset relative to the executing directory')
    parser.add_argument('--target-path', nargs='?', const=1, metavar='/path/to/target/dir',
                        default='new_data/leaks',
                        help='Directory where the data will reside, relative to \'darknet.exe\'')
    parser.add_argument('--nbFolds', nargs='?', const=1, type=int, metavar='number of folds', default=5, 
                        help='Number of folds for cross-validation. If nbFolds<=2, holdout \
                        cross-validation is performed with a 80/20 split.')
    args = parser.parse_args()

    print("Dataset: ", args.dataset)
    print("Target directory: ", args.target_path)
    print("Number of folds: ", args.nbFolds)
    #image_dir = os.path.join(current_dir, 'Images/leaks')
    #gdrive_path = 'new_data/leaks/' # Directory where the data will reside, relative to 'darknet.exe'
    target_dir = os.path.dirname(args.target_path)

    images = np.array([os.path.join(args.target_path, filename) for filename in os.listdir(args.dataset) if filename.endswith('.jpg')])

    if args.nbFolds <= 2:
        print("Performing holdout cross-validation")
        import random
        random.seed(1)
        random.shuffle(images)

        split = 0.2 # Proportion of the dataset that will be exclusively for validation

        with open(os.path.join(current_dir, target_dir, 'train.txt'), 'w') as file_train:
            file_train.writelines('%s\n' % image for image in images[int(split*len(images)):])

        with open(os.path.join(current_dir, target_dir, 'test.txt'), 'w') as file_test:    
            file_test.writelines('%s\n' % image for image in images[:int(split*len(images))])
    else:
        from sklearn.model_selection import KFold
        # Number of folds for cross-validation
        kfold = KFold(args.nbFolds, True, 1)        

        for i, (train, test) in enumerate(kfold.split(images)):
            # Create and/or truncate train.txt and test.txt
            with open(os.path.join(current_dir, 'train{0}.txt'.format(i)), 'w') as file_train:
                file_train.writelines('%s\n' % image for image in images[train])

            with open(os.path.join(current_dir, 'test{0}.txt'.format(i)), 'w') as file_test:    
                file_test.writelines('%s\n' % image for image in images[test])

            with open(os.path.join(current_dir, 'obj{0}.data'.format(i)), 'w') as file_data:
                file_data.write('classes = 1\ntrain = {0}/{1}\nvalid = {0}/{2}\nnames = {0}/obj.names\n'
                                'backup = {0}/backup/fold{3}'.format(target_dir, 'train{0}.txt'.format(i),
                                'test{0}.txt'.format(i), i))

    print('Done!')
