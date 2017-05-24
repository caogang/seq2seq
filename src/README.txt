Run python train.py to train orginal seq2seq model which is saved in ../snapshots/seq2seq_newdata-{seq2seq_epoch}.params

Run python discriminator/DiscriminatorDataGenerator.py --load {seq2seq_epoch} to generator training data for Discriminator

Run python discriminator/HierarchyDiscriminator.py --load {seq2seq_epoch} to train Discriminator model which is saved in ../snapshots/discriminator-new-optimizer-{disc_epoch}.params

mv ../snapshots/seq2seq_newdata-{seq2seq_epoch}.params ../snapshots/policy_gradient_g-0001.params
mv ../snapshots/discriminator-new-optimizer-{disc_epoch} ../snapshots/policy_gradient_d-0001.params

Run python adversarial_training.py --loadEpochAdv 1 --epochRL 50000 --saveEveryRL 1000 to perform adversarial_training, params is saved in ../snapshots/policy_gradient_g-{adv_epoch}.params and ../snapshots/policy_gradient_d-{adv_epoch}.params

Run python test.py --loadEpochAdv {testAdvEpoch} to show the result
