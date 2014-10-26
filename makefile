construct:
	python knbc_to_xml.py knbc-train.xml knbc-test.xml knbc-reference.xml

basic:
	python hmm_segmenter.py knbc-train.xml knbc-test.xml knbc-hmm.xml

evaluate:
	python evaluation.py knbc-hmm.xml knbc-reference.xml
	@echo Initial results:
	@echo Avg Precision 0.904005681695
	@echo Avg Recall 0.881382517888
	@echo Avg f-measure 0.892550767507

trigram:
	python hmm_segmenter_perso_trigram.py knbc-train.xml knbc-test.xml knbc-hmm.xml

backoff:
	python hmm_segmenter_perso_backoff.py knbc-train.xml knbc-test.xml knbc-hmm.xml
	
reversebackoff:
	python hmm_segmenter_perso_reverse_backoff.py knbc-train.xml knbc-test.xml knbc-hmm.xml

alphabet:
	python hmm_segmenter_perso_alphabet.py knbc-train.xml knbc-test.xml knbc-hmm.xml
	
total:
	python hmm_segmenter_perso_total.py knbc-train.xml knbc-test.xml knbc-hmm.xml

proba:
	python hmm_segmenter_perso_proba.py knbc-train.xml knbc-test.xml knbc-hmm.xml
