module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      "tier_capacities" = array<i64: 65536, 65536, 65536>,
      "alias_check_length" = 4096 : i64,
      "reuse_guard" = 100000 : i64,
      "num_semaphore_pairs" = 32 : i64,
      "num_semaphore_generations" = 4 : i64,
      "bus_size" = 64 : i64,
      "riscv_sem_base" = 65536 : i64
    >
  >
} {
  eaac.schedule @schedule {
    eaac.hw_alloc @unit0 : !eaac.gemm<2 32>
    eaac.hw_alloc @unit1 : !eaac.risc<1 32>
  }

  func.func @main(%0: tensor<16x16xi8>, %1: tensor<16x16xi8>, %2: tensor<16x16xi8>, %3: tensor<16x16xi8>, %4: tensor<16x16xi8>, %5: tensor<16x16xi8>, %6: tensor<16x16xi8>, %7: tensor<16x16xi8>, %8: tensor<16x16xi8>, %9: tensor<16x16xi8>, %10: tensor<16x16xi8>, %11: tensor<16x16xi8>) -> tensor<16x16xi8> {
    %c0_i32 = arith.constant 0 : i32
    %w0 = arith.constant dense<"0x390C8C7D7247342CD8100F2F6F770D65D670E58E0351D8AE8E4F6EAC342FC231B7B08716EB3FC12896B96223177494287733C28EE8BA53BDB56B8824577D53ECC28A70A61C7510A1CD89216CA16CFFCAEA4987477E86DBCCB97046FC2E18384E51D820C5C3EF80053A88AE3996DE50E801865B3698654EBF5200A5FA0939B99D7A1D7B282BF8234041F35487D86C669FCCBFE0E73D7E7320AD0A757003241E752210A924798EF86D43F27CF2D0613031DCB5D8D2EF1B321FCEAD377F6261E547D85D8EEC7F26E23219072F7955D0F8F66DCD1E54C201C787E892D8F94F61976F1D1FA01D19F4501D295F232278CE3D7E1429D6A18568A07A87CA4399EAA12504"> : tensor<16x16xi8>
    %acc0_0 = tensor.empty() : tensor<16x16xi32>
    %acc_0 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_0 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m0 = linalg.matmul ins(%0, %w0 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_0 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w1 = arith.constant dense<"0xEA33256D8743B2237DBD9150E09A04993544873B364F8B906BAF6887FA801A2FD88D1601AA428652E2DA0439264C12BD4BDC41159DBA14B76B7F34B5D04F79535AD30C5BAAD27F885137C313F07166EBB39C74720C62CCA88E238EB3CCA90E3B855B871337DEB0A0DF3BC5618216DF0064BADC23A9A03F999ED1A7CE974162D7C2599ACF009B926BDCA4EEE2E26DF2562B91AB2F789E73654B0C177DF325E9D463C4FDCC7C4B0236D9705AED197F3EE944EDA2E2DAE451F3E6847E8DF87A8CE12792788BABA329464D76C44E6D20D4D0A9EED41F69D7C70AC2F403B498C7D670F9708BDFF80EC7ACCF54EF410DC90D2ADB45EC5D1985C2A76CE8A7ACC28ED781"> : tensor<16x16xi8>
    %acc0_1 = tensor.empty() : tensor<16x16xi32>
    %acc_1 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_1 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m1 = linalg.matmul ins(%1, %w1 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_1 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w2 = arith.constant dense<"0x29F0091AB37223140F7E660A4E7A40F23A6FEE83BC553A539F370D9FC0CB65267C349A3D15B1DBBD23AE06D7FA36DDB9EB4EDE5A8AF7EEDF89A57D2C8EE67CEDC2AC0EFDA65DF96CB584AE8F8D05612B7BD0FA7BF3FBE5082F9671CF7C9CBCF2B0D9A9B4E88A9C80763D62A13D5E626EF78D9033639774B85B9A07408C171B9540FB340691F0F5E1AE5E1A81F43A21CDFB251B4D4C9B2B7F3CD573C2E6E298DB9C1E326A6C8729507A58265001D1E6F09510769390E824778765D93A734C8848241E549D93E03FEF9BCE8BFCE02914DDA5800D2E750A891459F0E28E5CDFFB2EF0B2D1AAA43552A8D2FD93CD12E82DA181A53BCE00ECD31B60B9FFE21A688843"> : tensor<16x16xi8>
    %acc0_2 = tensor.empty() : tensor<16x16xi32>
    %acc_2 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_2 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m2 = linalg.matmul ins(%2, %w2 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_2 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w3 = arith.constant dense<"0x93E0F83E0E7A519F07D02F733AEC3C4EFF958BD4F7F17CE94AC46145238DD4AE88019098FA4CE4F7B0AAC1E9A4607AC477D216A2F2C3C54DFD1240A933E133E90749D14F26F087ADCB29A8C2A2F912237893742EDE3233E355990E17A61C96B7BFDC4A7DD25C575928C37BFE4976EC82EB8204EE935025E2B099D980E99A65C4F73679C3B797970BCA8C0419FE9275B47061804631149EE111BA432E97A7D4596643BB8B5483F697AD3AEF264873CBBB2ECA07873FE8BC86C3BE3777F10CA77120ED9AD13B4717139BFC3B317845C6E8BDD64FD432FAD08F10BD6FE3E378B932BCB71FCB8D613EE82E6C0A19AA7C4069236A6E77A84B018D4A428059380D4307"> : tensor<16x16xi8>
    %acc0_3 = tensor.empty() : tensor<16x16xi32>
    %acc_3 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_3 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m3 = linalg.matmul ins(%3, %w3 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_3 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w4 = arith.constant dense<"0xB779A50859871A40D73A20F3E5B937E771169AEA0F1FF5CDDA37FBE32529A44B21408CA6C396E8DC323A6EDCE774D3ADE8CCD430A0DAA082BF4EF2222E2B2FDD31BE421EA83ED2B5D81A939FB4356C4FF67237B3BC3A8E73DB0D880E5C8B9EADB3035C49CD23480F2E6EC0D6E8AE50BD9FA62B1A4F5019298BE2D9F8E2D48B6E3AB0DC3891F99D1770CA1C03689A6C468294A73D03FEDC5942C275B524CB15DF09EB27A0DBCFD5943ACF0AA657EBB92DDF367CDFCD28CA9EAD71AA56273A63B2B34B78344A8365584E265AFCEDE5A5A14DE122F0E29B8C1CB4259EECE7131DBC92272EC4EC15E660A4F34D1FE634AF2B58147EE0E051BABE90C6D1AD1AAB21A8"> : tensor<16x16xi8>
    %acc0_4 = tensor.empty() : tensor<16x16xi32>
    %acc_4 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_4 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m4 = linalg.matmul ins(%4, %w4 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_4 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w5 = arith.constant dense<"0x30C591814CAA2948B39EC8422B9EC0A8412FD8B909B99E5C6DAEF86273464F27973313AC43C04E535C54E016D2BA79E391E5777A9EF063BCE1EC90C3D6526646801AF6BE343F912A528BE64BDF2E71E6B20DD41BCABF78C529BF720EA332AB4A461392F147F0E5022809836E4CD83893799A3E187AD6EA2038FF087B4995DB00B47BD55F2BB8220AC7F016C6BF8108B622B07B35AA4416B4AD59EDF55D4520EA129667166615A19ECBF281126192B618A98B3FBCDFCCE1C5AD5FFEFEBC882AD928DC5C96A43428A7979CE4DA55E3B3E415B4DE8C1D26CFBA510F49E011402278BBB9C4104EE6BDBEE32746BBCBA08E7F3A0D5FFFC63C8685E46D92FB663E4525"> : tensor<16x16xi8>
    %acc0_5 = tensor.empty() : tensor<16x16xi32>
    %acc_5 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_5 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m5 = linalg.matmul ins(%5, %w5 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_5 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w6 = arith.constant dense<"0xE758E32CA3B12194995059B9723E664779FC0DB8BCEF422C219ECBF5D2D12540A225E6EEB0415D42DD1C3F4E9B5452A573B1912880648C409B2F564E57AC150E2917876BD50FFE949AF77DCF98E8251E50E1D4F7ED68AE49A0A3B0CC42BD36A37BEE3E88E67E48311994C4D67F51A7A06151FFEFFF9DFE0B2EC9EA7B6EB4181990FDF0920437DC4487BBCEBB17CD1A63B99325C5E68F3C4131C9BFADBB4965CD14171346AAF2E94C47A7A353C999ACFA99F308BCA938D59D0DF287741AF557C24B7C10386109E1A0D64DD368D2F11F466AA6F4C0A058EBAFB587F7627E8E987398936AFAA2F5B28C933EC2CAB04A94159328B1E283F56D678A8B46377A7C1973"> : tensor<16x16xi8>
    %acc0_6 = tensor.empty() : tensor<16x16xi32>
    %acc_6 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_6 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m6 = linalg.matmul ins(%6, %w6 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_6 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w7 = arith.constant dense<"0x771A33D3A9F133460250D0F3F46693A4921E2D761359D55A12CBFD5F9413049836AB91E8FC44EF8B6239A953EA835F07AC976259CFDAA72CCD305E47F4A57F0385C478E488A89A0585B8781F3CEE9D51CF9F3C97BC717044F44EE8BFD4F16F7E29E4B927391F674C54A7E23B69FA2EE41CE843D4E91DEC9D0BCA82016F2517D8B0201E23F11092D15C45D7BFC3E5C1C02944B23C5BC94172010B98EDD9C2757EEBB14F8D603910D6087B69223311E4187D16CDE0776F1C479477A3A4799A4971D3998C1F59DAFD18B0C3A3D5D14C99C05EF27B739949ED1DD3D544C67C8268A928E6BD2F611A89C11425606FF56AAA9B076C613CF57C68CB7AA490C2EEB79D85"> : tensor<16x16xi8>
    %acc0_7 = tensor.empty() : tensor<16x16xi32>
    %acc_7 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_7 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m7 = linalg.matmul ins(%7, %w7 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_7 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w8 = arith.constant dense<"0xB8FEEE32F0A368BDA0D317714A0885D5974E64A875C27DFFAC83FAFBEB56B45647FA5E1E11261803D34676224D046FE9BF1EF7F90803D206088C9208DC5B36314C7B6281B588CB28BFCFEB7C739929102FCFC2C1F31C04572AFFDEA93015756CF38A17268F105BA1086A49CB2799537BC7A9C44728B11B32DF7626AECBA70F8BE6FB74B6C0DD5FC22B977E252A894EC24EC7A2B8362E029DE3B88A34432C5FDCE5D0340D2DB52FA6C50695D3C62B7C56C25647899A89FC4A2055DE8DD799F727B8807EFD64EA36459B03CAAAC2A8E1ABDC4599A466F5A05ACBA395FB7CA6C08FC9BA3A665C0DEC6BE09523D1FF479B7B814ED8C125E5F5CDD612B82B377FB555"> : tensor<16x16xi8>
    %acc0_8 = tensor.empty() : tensor<16x16xi32>
    %acc_8 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_8 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m8 = linalg.matmul ins(%8, %w8 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_8 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w9 = arith.constant dense<"0x16CCA9DC360532847171E4BFC8ED4DB00CF73597D42B3B48B29FAFE969F7B2F331E0E7A32299163A0BAF37547C5951A9DAEC76CF5E5FDDCA0E65E6DBC7026D698E20345FBBA664EA3A86FAA0C6C83AB2B4EA58982B44A03C7A9C3B5DBF48C6D646C4D85FF95855FA93475FA1E61BB704F84563C4FDD1FBD4E3FA552A0F7095108C739356EAFD393A89BB15E16FD9347E9810E686B22CE03C796BB3DB544769691EB38F56A59594883045D21E8D40437F4AA47EC9FA4889D4C0E7262FCE8EBCE8F9A7012FEAB720CB6FDB6CFD89A591AC42F8AF181732EB083F50E1E900DB67439A518C2FB8802ABE541ACA9C77DB2E30006DF4274373E30404AF3DD843F42475"> : tensor<16x16xi8>
    %acc0_9 = tensor.empty() : tensor<16x16xi32>
    %acc_9 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_9 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m9 = linalg.matmul ins(%9, %w9 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_9 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w10 = arith.constant dense<"0xC42D3434A0BC9946C34449230454E1B36D4DD2E26F2C33473FC4B3DBA1477E8D2B7F910D9A6960C8971B7AFDC5397BFF2406B8A243C6D7BB58F125082207866E141ECB92D4D8CD2A4E8E2A9E28684FA7C8219EDF7A1D7D2CDE3BE81C9E593D06460553FEB18455BE40893C0FABDB8B208627FEE9B81CFF55BC5082343B740116067D17F1BAC44C5B12D672A47FD5A38A27BE3D1A5B7217CD23EE909FA72CE904BC66959B7CEDBCFC647D4D03D10C77B104AB00C09D356979D6FB1E488F2F1676D3BEEA2B3044C9261FDE42799585A1C9A7A2E78B77266745374F3654E6EEA0D03DB76AE79DED873D2E509B146EA74B2E7FB6CA199985590FCFE77F30EC34473E"> : tensor<16x16xi8>
    %acc0_10 = tensor.empty() : tensor<16x16xi32>
    %acc_10 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_10 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m10 = linalg.matmul ins(%10, %w10 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_10 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w11 = arith.constant dense<"0x061F714265CDBE2B84260B2165E3412FA93E15EC1956DDCAFE0FC3DA58B56D5F8C8FE44C117D97FFD2F51F2C8FC446D6667F09C3B7F5F8B0A4C68A5C0DA3700F8F1DF1B77751337E7B881C70C6B5585A79A2B70EB44860FC9E59FB132E1C77700AF400A96742AE5AA51E0B4B4838BA25BFCA33AC9AA54550DFF9A259B67259C19D96415A00C8105DA27135FE48A92779B1A3552DABE405876B7F22B383370018C5E0D655D3FCC1B3C033F57353E72511960AA3853525AF57C052262FADF70FDC54DE501B33A96961D1879098793119C9FB4E1BB801DA2C96F566310D69579529F23B9DCBF1FA872EC65EBDC3BD5FE41686E1EE8673891F4D312BB0D3741FC6D6"> : tensor<16x16xi8>
    %acc0_11 = tensor.empty() : tensor<16x16xi32>
    %acc_11 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_11 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m11 = linalg.matmul ins(%11, %w11 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_11 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %sum_out_1 = tensor.empty() : tensor<16x16xi32>
    %sum_1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%m0, %m1 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_1 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_2 = tensor.empty() : tensor<16x16xi32>
    %sum_2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_1, %m2 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_2 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_3 = tensor.empty() : tensor<16x16xi32>
    %sum_3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_2, %m3 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_3 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_4 = tensor.empty() : tensor<16x16xi32>
    %sum_4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_3, %m4 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_4 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_5 = tensor.empty() : tensor<16x16xi32>
    %sum_5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_4, %m5 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_5 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_6 = tensor.empty() : tensor<16x16xi32>
    %sum_6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_5, %m6 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_6 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_7 = tensor.empty() : tensor<16x16xi32>
    %sum_7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_6, %m7 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_7 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_8 = tensor.empty() : tensor<16x16xi32>
    %sum_8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_7, %m8 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_8 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_9 = tensor.empty() : tensor<16x16xi32>
    %sum_9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_8, %m9 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_9 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_10 = tensor.empty() : tensor<16x16xi32>
    %sum_10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_9, %m10 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_10 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_11 = tensor.empty() : tensor<16x16xi32>
    %sum_11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_10, %m11 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_11 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %bias = arith.constant dense<"0xF378F39928CA114B3DE558556C624015D828DE674B84A3222CC6CBA68DEA06D639D44C4B33363392B0D287C4F8F213588DCE49CD13CAA97919F389BE0AAC9B9F"> : tensor<16xi32>
    %biased_out = tensor.empty() : tensor<16x16xi32>
    %biased = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_11, %bias : tensor<16x16xi32>, tensor<16xi32>) outs(%biased_out : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %relu_out = tensor.empty() : tensor<16x16xi32>
    %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%biased : tensor<16x16xi32>) outs(%relu_out : tensor<16x16xi32>) {
    ^bb0(%x: i32, %o: i32):
      %rc0 = arith.constant 0 : i32
      %cond = arith.cmpi sgt, %x, %rc0 : i32
      %r = arith.select %cond, %x, %rc0 : i32
      linalg.yield %r : i32
    } -> tensor<16x16xi32>
    %quant_out = tensor.empty() : tensor<16x16xi8>
    %quant = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%relu : tensor<16x16xi32>) outs(%quant_out : tensor<16x16xi8>) {
    ^bb0(%in: i32, %out: i8):
      %M0 = arith.constant 1091133696 : i64
      %rsh = arith.constant 35 : i64
      %nudge = arith.constant 17179869184 : i64
      %zp = arith.constant -128 : i64
      %lo = arith.constant -128 : i64
      %hi = arith.constant 127 : i64
      %x64 = arith.extsi %in : i32 to i64
      %prod = arith.muli %x64, %M0 : i64
      %rndd = arith.addi %prod, %nudge : i64
      %shft = arith.shrsi %rndd, %rsh : i64
      %bzp = arith.addi %shft, %zp : i64
      %locnd = arith.cmpi sgt, %bzp, %lo : i64
      %clo = arith.select %locnd, %bzp, %lo : i64
      %hicnd = arith.cmpi slt, %clo, %hi : i64
      %chi = arith.select %hicnd, %clo, %hi : i64
      %trunc = arith.trunci %chi : i64 to i8
      linalg.yield %trunc : i8
    } -> tensor<16x16xi8>
    return %quant : tensor<16x16xi8>
  }
}
