#!/bin/sh
echo -e "Pretrained results"
echo -e "ATR"
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/atr_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/librispeech_pretrained_v2.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
echo -e "maono"
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/maono_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/librispeech_pretrained_v2.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
echo -e "clipon"
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/clipon_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/librispeech_pretrained_v2.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
echo -e "USB"
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/USB_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/librispeech_pretrained_v2.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
echo -e "USBplug"
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/USBplug_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/librispeech_pretrained_v2.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;

echo -e "PhyAug"
echo -e "ATR"
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/atr_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/meetingroom/deepspeech_meetingroom_PhyAug.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
echo -e "maono"
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/maono_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/meetingroom/deepspeech_meetingroom_PhyAug.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
echo -e "clipon"
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/clipon_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/meetingroom/deepspeech_meetingroom_PhyAug.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
echo -e "USB"
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/USB_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/meetingroom/deepspeech_meetingroom_PhyAug.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
echo -e "USBplug"
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/USBplug_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/meetingroom/deepspeech_meetingroom_PhyAug.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;

