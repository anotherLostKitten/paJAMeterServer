sequential: binary_classifier.c load_data.c
	gcc -o sequential binary_classifier.c load_data.c -lm -g

clean:
	rm -rf *~* a.out sequential *.jxe
