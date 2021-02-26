python detector.py --epochs 100 --save-directory none_nobn.dir --no-bn --rotation_type none
python detector.py --epochs 100 --save-directory allangle_nobn.dir --no-bn --rotation_type allangle
python detector.py --epochs 100 --save-directory flip_nobn.dir --no-bn --rotation_type flip

python detector.py --epochs 100 --save-directory none.dir --rotation_type none
python detector.py --epochs 100 --save-directory allangle.dir --rotation_type allangle
python detector.py --epochs 100 --save-directory flip.dir --rotation_type flip
