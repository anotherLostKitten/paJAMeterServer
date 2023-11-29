sequential: binary_classifier.c load_data.c
	gcc -o sequential binary_classifier.c load_data.c -lm

clean:
	rm -rf *~* a.out sequential *.jxe
