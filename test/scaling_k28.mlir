module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      "tier_capacities" = array<i64: 65536, 65536, 65536>,
      "alias_check_length" = 4096 : i64,
      "reuse_guard" = 10 : i64,
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

  func.func @main(%0: tensor<16x16xi8>, %1: tensor<16x16xi8>, %2: tensor<16x16xi8>, %3: tensor<16x16xi8>, %4: tensor<16x16xi8>, %5: tensor<16x16xi8>, %6: tensor<16x16xi8>, %7: tensor<16x16xi8>, %8: tensor<16x16xi8>, %9: tensor<16x16xi8>, %10: tensor<16x16xi8>, %11: tensor<16x16xi8>, %12: tensor<16x16xi8>, %13: tensor<16x16xi8>, %14: tensor<16x16xi8>, %15: tensor<16x16xi8>, %16: tensor<16x16xi8>, %17: tensor<16x16xi8>, %18: tensor<16x16xi8>, %19: tensor<16x16xi8>, %20: tensor<16x16xi8>, %21: tensor<16x16xi8>, %22: tensor<16x16xi8>, %23: tensor<16x16xi8>, %24: tensor<16x16xi8>, %25: tensor<16x16xi8>, %26: tensor<16x16xi8>, %27: tensor<16x16xi8>) -> tensor<16x16xi8> {
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
    %w12 = arith.constant dense<"0xF378F39928CA114B3DE558556C624015D828DE674B84A3222CC6CBA68DEA06D639D44C4B33363392B0D287C4F8F213588DCE49CD13CAA97919F389BE0AAC9B9F8FFB32744499E2A489D52D60E26CD0F8BF1C51219FCE450E5862661D7F10E919B8658CBCEDCB3F0F7BBEFAE45AF2B3B053842E910EC519536D71736989D10A04F243580472819C8CD8C0B2EB816FEE9ACB3403BF909837F620AC8DA68D8598604C7B1DCEA3470DFE9784D6CFC61261AF71F3B799575C9631F141845DAB2D71B5729DD7A9BE8596ED3CF01B24F1633DC398D31B4D4666AFD1EA49A15C29F9AA5AA01D02E78C6B5651FA2D42DEDBCEDBF4C4021366BF07A46109027F73B09F4234"> : tensor<16x16xi8>
    %acc0_12 = tensor.empty() : tensor<16x16xi32>
    %acc_12 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_12 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m12 = linalg.matmul ins(%12, %w12 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_12 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w13 = arith.constant dense<"0xC49C5521179894E9ACDE44AFFAB76054CE097472436C0B563FBB13C0821818370A183BD5E3C23F81F24E69049AD7338A46D6353C903A36FE65668667B7D1975214FE6AF6A97801062D38FC4C2E26338376E99787ED1933591494B8A0DA3B3017044657A8B7E0892FBEAF5D3ACDCCE98BC5F4D75539431E5135D5F7E05CC2B70E42F7F939D2E1172085A00672AD30DD7FF6B1C54C1B0550F0FD5426FBA47BA98E1972C1CE7828EBE3E52DFFE4A13CFE0934CED312002E9B69E9AEBA1B72E8A9F2BAE73A317101AEB595BF361754FC47D1368264603DC96EEA61AD37D4173FE6EA44FF0218DDF75A584135C6A4C4D67F8CCBAF96E74446D399A06F666B94B0435B"> : tensor<16x16xi8>
    %acc0_13 = tensor.empty() : tensor<16x16xi32>
    %acc_13 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_13 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m13 = linalg.matmul ins(%13, %w13 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_13 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w14 = arith.constant dense<"0xA53BB4F5D592D806360CC34A1B1C66885292804A1F976810B7E73472C7799318C3C6D5A01B068067E273BC646396E55A265F583DC114DB8E8742528202A9EE4F144EA51A9AFBB725A2705E2251D4CE2FB2716DA9A9B8966FF3043AB1E47B7C14A4C33AC583900CBFE4F9169762A42A315503216B6DD93569DE244E0CEBA9132C251B5B80247484D0C6E6CFDDA108C73C03201725B736949A2C90E0C3CA0FF2507045C8974998BF05483F1502C92B9C6BB06CD348525E7181633A5C1BEC259422803265F9A9B6417833912363A2F8E6A89C4EBAA1DB5602A07E71DF8DB844A9F5ECE4B79CF93759298F4764862726090FCF6C1285F14CBBC8729444ED2ECBBA05"> : tensor<16x16xi8>
    %acc0_14 = tensor.empty() : tensor<16x16xi32>
    %acc_14 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_14 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m14 = linalg.matmul ins(%14, %w14 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_14 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w15 = arith.constant dense<"0x7A5447E4515B4AF7B71470F9762087BD771569B8C6F0E81714A5348C8659C0C1BB2780C072CFB4B0F9F60347E15475278F6C4C5F50B93C75D4A93AF6F5F56B52D20B7C16415244144B1C55845CCF09912B6EE5F1F65770D2495D8755AFE82375C1C543370867D64C315AF3133C2FF64200DCD7BD16DB75F6C3B332D94482F4772A91D1836002033C03C069A1D1B9364FDA8185D8BB8DCB68D3F75BBDF7819DB0CB463C75E452336DCDAD1A52C7022E4AF1DE2D208671BD2813C4CD921D59FDE00567926311765612B87A00502C8DCCB4CACF6607232E91B2B18D7363E8CE028559D0362596AE3B8380E2C1F77CADEAEB0EF63B8422C37075A1D3B404867F932F"> : tensor<16x16xi8>
    %acc0_15 = tensor.empty() : tensor<16x16xi32>
    %acc_15 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_15 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m15 = linalg.matmul ins(%15, %w15 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_15 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w16 = arith.constant dense<"0xCCAF265BF3A0B56F895D61FF66787B876975677FB64F363D28FC0315F6BEA1FC8CFB5D8FCB71411BD8F410B1C959EA3DEBA104066BC233A6B9845F867989F1AFBB5EFFD96BD6BEBACD8FE65C4375338A78DDC3644A4CF2045AD7FF44A1F9998018CB32531E6E82F331E6AB85B8B883B00AD23EB3655991B5A6F9E428C392ED4A55ACD7CF242852AF74A0A29588CA8AE3B9E6D3555E0B407880286B55CB343020F21B0DCB2A35864F2BC79E757E91E24043570E11B6A6F3EF8BE3455AE9DE9EB1F6742DE29DBBD3854F9B050045B092F004FBF49904DC957603BF60D2DFCA7F53C7C1708628D87A8B9189D360235D459438E9E78BFB60A4094D190B5F34908940"> : tensor<16x16xi8>
    %acc0_16 = tensor.empty() : tensor<16x16xi32>
    %acc_16 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_16 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m16 = linalg.matmul ins(%16, %w16 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_16 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w17 = arith.constant dense<"0xE004772E4D07FC6FAFD79FFFBEED10A9294A1C88C824F05C6280C3063E7FC5E27E1165EB306AF2C79FA551153E81F69865B81A6F3978DEA607558D3DCA7C0415F34FBE2D7F50FF2EAD1E1EB64E3640F977A2B1E0620AC1B7448014F8E096FA615FB2A5EF8C52234FE69786BBE800C25E09BA7AA420C661FE4E90A5D1A438312216425B106736F8623210C6FA8319BB7124066AFB28EF1FB8D4BC1D8814249B276E313376C561627F5BD436B7EADA35555CCDAC50E2B4A1A6BED3DCA57B6DC082B8E7E5CB00119108459A0399716DA10B8D4F3013EBB64F4687BB7F7783992FCC6F27A6F758F9309913C3281CABB2DED53D7405C31B9C15A34CA7714E19C9E42E"> : tensor<16x16xi8>
    %acc0_17 = tensor.empty() : tensor<16x16xi32>
    %acc_17 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_17 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m17 = linalg.matmul ins(%17, %w17 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_17 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w18 = arith.constant dense<"0x71DD94C94B43AC2AF319CC862BCCDE4245F665F9CA647E7C2488DCA52D39D3D8BBB261A4B00C9C56B3E24AE1167766BF4F2B391B7C4B9307AF389CE6F10BA96B6C6884F74809C01682571E91668A1BEE0422C68AB8370995C576C232A1F82D5F65EA0B147E9CAAFAC95009B1B4B59A788D1F83C60F8F734DAEC9462CC374D57B4EDA9FA8E543694667469E45DC4A50A2223C5EA24A0FA3D3D29BE28C4D6036435E0D887676760DB728F444F0BF5EC6FBF24653627310D1078AD3716768215AE0FAA6EA7EDACF15C2A0C91337E7E9157DCA14C6196195E1CB9B92EC55D8961B48EDE72E395C45C9D3A5E60C5BE867E82E80A5533F2E3AE6B92BC451C1C21657AB"> : tensor<16x16xi8>
    %acc0_18 = tensor.empty() : tensor<16x16xi32>
    %acc_18 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_18 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m18 = linalg.matmul ins(%18, %w18 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_18 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w19 = arith.constant dense<"0x4C534A5B1AD69D9EE4935C3C4367018106EAFEFF02FC796FF3E8D1E228037D8E6A195E4B1CB2124A1098A92C1937D55B52915AE9B8122F99A5BCC222DA2195F1443E8CA8D36B6B1D372CF793DAA284B542FEFEFC5C779463003C0CEE3C6AD4D317AAB36C05679082229D99A5033AAAAECA0E5963A9ADF5F21245F64FEBABECD31C99A1096AA9B5214C8F0FADC054C75F3B9DA1108BD32CFE79BEE79A6E9A5939BF13672FE30CF74E61AEF79EA74932436912EBC63B029F162CAAEF17349F0B9F6BA46662928DA3E13C8B4C508C3A073469B3A38A48917D57249FADE319420EF11936C41B2CDAC85BE4B69FAA6528EE432FEF31E99EEEF076724984BAC85DA9A6"> : tensor<16x16xi8>
    %acc0_19 = tensor.empty() : tensor<16x16xi32>
    %acc_19 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_19 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m19 = linalg.matmul ins(%19, %w19 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_19 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w20 = arith.constant dense<"0xE8F90F21F339B3F5EC5790B24134CF36A6249AECCDD37A7C51F6C421AE6ADCDEAF785827F93AEDA631E35A9DCF2337C2A7FD1F2E6F47004B4CE795C7E46469A2999BDEB02D04CE41961FBD1B00B738900F78DDC9A2C3CFC3DE6C58F34760C4B8D54E93404405240849F9DF452B46D4E9D95CE8D9E6C4B5735FD903FF7989D2BB1113F7BC39F628488FEE282D7DF1279F1BAF05F0E8F0A9D7B4969455129B851FF2561A950D92F82A40278E81438F8C830739594DE4BD598578F85C10AF2B4A4EFDD3C479E1556BB01B873B1A55EDCCF39AF377C2C7F2864ADE0693E5C3A11531B5DD8BAB8A883282C8EC7B63AC5CA005042206D23235E4B370918062110C2F1D"> : tensor<16x16xi8>
    %acc0_20 = tensor.empty() : tensor<16x16xi32>
    %acc_20 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_20 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m20 = linalg.matmul ins(%20, %w20 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_20 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w21 = arith.constant dense<"0x6F3E3D2C42C05E385E8196EF26E88DDAA58352258D1FC3C6AA1E84875BE123FB66C95BFB3749DC6F36E2843411252480E409FE4C0013C7884DA8C801FBF4DFCE237301C4F63EB7AA9BCD1C2EBBAE29EB94675A14C46199C736B7F92BE0614712620C7AFAC7209DFFE60812AD1CAC15CF3C6CEA4C77241248FA09F1177C420758199D31E95720363AFE702067DAACCDA63D46A46D1758AB203D8C48D456BD54E07C144433C05008F92903035B2D91D5F34069498AE6D27A59135CFF2D9AB2D60278623C30B970ED6A8D1B63394CF2FE5A51161BBA916BDE80ABE816351A5A500EF83B2478E3A6D665A67538A6743279D366C07025F1F81BDF4A6EE85BC6BF45EA"> : tensor<16x16xi8>
    %acc0_21 = tensor.empty() : tensor<16x16xi32>
    %acc_21 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_21 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m21 = linalg.matmul ins(%21, %w21 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_21 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w22 = arith.constant dense<"0x63B78A67AB41101E8F4538EEEAD657AF70975057752CD2FA6EF2EC8E07DB7E7EA7DE714C02D51614F011D9C9F532E7FD7B2A06D06F43B7977A2E08F59C52EB0CE8ED46CF5EBDED0024C5AA1097DA8A6A29AB1CBA1A38C4BBEE14BBBB59226846AB7A3AF46B73A4B488D88D1B20BEB0E7F83EF095E7BC2160EA2C29C85B932737548D394BFE4D680B96D543F2B1D7E30A9560200B76ABD2AFA7F8E3EA9B439C578574F7BE93184BB491E95552925D594E52793B15A240E185BD5E6911A2B04BBEE5539A3E0CCFFF4AEAA34F02A997F2944BE2AD39F514473B9D0CCD70E4B62FA71C63718E640B2AE42CC963FBA21CB8CB4BDABDC5BE0691B0B39DE42C6EE3D4E2"> : tensor<16x16xi8>
    %acc0_22 = tensor.empty() : tensor<16x16xi32>
    %acc_22 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_22 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m22 = linalg.matmul ins(%22, %w22 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_22 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w23 = arith.constant dense<"0x9353D9D4472C496A9BF3AB21C02D03D18BD459EC46665FC394FB159A9BFCED720D014E8699F13F93D55BFB7D0E3D8DFC22B9F6DF84F33DBF0884531D4AA3EC45B6531B8BB8D6B618949527635D3055A5A19A8AB931B449064690BEF8D2AE53346F4F71603AD5EB273B4308F5B29FB947E2AF589B0E3C714148749D406B4228D9392709F7DC156437101EF03C8116219A05313F69665EB30E2E4A8EB2B4527B2232076F62905702326DF3BB5EE1365CB5C2D96D74D039A03FEE3845FDB893142AA7FC3BC591E99FE373D094DB6023F3BC5B06FB35CAE02B3D49CACD345621BA65AD66CE3E77D9FF8E8BCCB6F3CFF88E4BE8373E80FAFE76440F321B22FF0D5BC3"> : tensor<16x16xi8>
    %acc0_23 = tensor.empty() : tensor<16x16xi32>
    %acc_23 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_23 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m23 = linalg.matmul ins(%23, %w23 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_23 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w24 = arith.constant dense<"0xA74E9B3127FCEE6796FEFB85AF0910DC3CC4E6A1C0E2EB12105A9B54FBDC449FC7F938FBF0F3501CC886FDC1B491A41F0E1741BBB20DAEC231DBB0B6D2B31C7D7E789F2B219C3AAC560AA613EBD8D7880AF00BA2C1C7A532FA3369ACFEAD5599BADF67FE266D93871898CD732F30AC8B22F3DA6BB1279A99DDA066D4CD80E85C095FFF6DEF89DE1E66BAEC4B7B4D99DED2A32068D38892528378CF7A3A4C620E7D9EEDF8DFD81267518B450E848E5A15F80A1C2AA09820D0BA2ECB4778393D80FE2E793EA0A8C778D86F4936596D8CE1D1821CFBBF39A2D8F37E8E4AB9F3BC9D0A111B7A81C49EB93E79E2C604C9829A2E78986B52E0A2D0118C29FBF1D3AAD8"> : tensor<16x16xi8>
    %acc0_24 = tensor.empty() : tensor<16x16xi32>
    %acc_24 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_24 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m24 = linalg.matmul ins(%24, %w24 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_24 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w25 = arith.constant dense<"0x33528521B4D7A7E80E1A5EE34501CD669CA02A38507F42D2A3E7F4448558A790652BD4BC5DC9C1720A0B752DEC65865619CDDB854C2DBC573C07DEE8FD3C7268618893FDC342DC696E9060D65CD14A0E948678E67FBDEA2B05F044A5CB6CC4D88325301E1AAA78F1B5170281C39ACABD43155B8556FCADEF432DA2DD48853BA01641FF8FB8DC908687A0DF54A1FB4DE11B604C1B755D12F08553EFDF3BFD14E2B6F8EAFC394B3FB22F68915C568B6C025CD5277EFF30A80D913EC9BF36A54E0E29EBF5E88902A816E7A6426C1890A51C24C32FCC825B7ACC2D0A6A53C6CE914E3D98FBDD9E43B731BC66FAE4C6F389D82ED8BC5FDEDA2F14AEED724FFD2A94EF"> : tensor<16x16xi8>
    %acc0_25 = tensor.empty() : tensor<16x16xi32>
    %acc_25 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_25 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m25 = linalg.matmul ins(%25, %w25 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_25 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w26 = arith.constant dense<"0xDF28F4994C41997FDA9718271737D9D330ED81B58229083AEC5B5DC9172AD598A0E8E687AE5F6C5E288436DE0902CA3B0657F4EBCFF99BB8CA9A2C404507613908600D6BF64A7BDF612C9AED4BF2D80B3B554EE3F8F10AE67C6796C81AAA0434DDD4FD9412F9CCF8F9761F09D8A12089D5F23BFDB1E5DF0A529362F0B0E07670ADEF4396C7D5B8F1526A902CF742586650FB33D4FC9B786B1809B785B651591020ACDC3A743D72F984AEB517884D3519E0EA3836341535553C69B95095E2FF324EBAAB463E67E0D4037DF2AE92572B88BED4B87D4DD77446DBD3D1EAB90FE24A05232BADA1CD54DE1FC9124D1D97D6CD9BA0B0048F4BCEE469E2B0CF659A2B12"> : tensor<16x16xi8>
    %acc0_26 = tensor.empty() : tensor<16x16xi32>
    %acc_26 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_26 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m26 = linalg.matmul ins(%26, %w26 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_26 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %w27 = arith.constant dense<"0xABC740916D2F9FAE510B0E847B9BD8381EBE84B53F183FE59802E8AD7877B3999147C17E35A762B8E8324BD9D2292F1FD942F36DE427BDCECA87FFD8E1FAD5534941F8EA423C6CE9F21050BD3CD1383DBD94144E3CBE9CBF9B96F80E4948D7189C916E5DFA013EB22216D5F69EE7ACC1F34346EBE6E54CA62DD1C677CAAD17E28E2D540C9E3F2B621B60F3093D5C098C697DE32BA4CD17FB0BAEDF5CD40CA2C3832E3A325692495D822D4E9DD0561EA65C0D1D98C732A348D3A584C03D1A677A79FCC53A891FA2EDD021F70ED0770989337E409BD0E42A6848BF567DD8E71500ED9455001EF68B80C2B36C90175223992DCA3E478FDD6BA2AF85E78B7DA4AE6B"> : tensor<16x16xi8>
    %acc0_27 = tensor.empty() : tensor<16x16xi32>
    %acc_27 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_27 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m27 = linalg.matmul ins(%27, %w27 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_27 : tensor<16x16xi32>) -> tensor<16x16xi32>
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
    %sum_out_12 = tensor.empty() : tensor<16x16xi32>
    %sum_12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_11, %m12 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_12 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_13 = tensor.empty() : tensor<16x16xi32>
    %sum_13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_12, %m13 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_13 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_14 = tensor.empty() : tensor<16x16xi32>
    %sum_14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_13, %m14 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_14 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_15 = tensor.empty() : tensor<16x16xi32>
    %sum_15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_14, %m15 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_15 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_16 = tensor.empty() : tensor<16x16xi32>
    %sum_16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_15, %m16 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_16 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_17 = tensor.empty() : tensor<16x16xi32>
    %sum_17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_16, %m17 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_17 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_18 = tensor.empty() : tensor<16x16xi32>
    %sum_18 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_17, %m18 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_18 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_19 = tensor.empty() : tensor<16x16xi32>
    %sum_19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_18, %m19 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_19 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_20 = tensor.empty() : tensor<16x16xi32>
    %sum_20 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_19, %m20 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_20 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_21 = tensor.empty() : tensor<16x16xi32>
    %sum_21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_20, %m21 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_21 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_22 = tensor.empty() : tensor<16x16xi32>
    %sum_22 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_21, %m22 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_22 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_23 = tensor.empty() : tensor<16x16xi32>
    %sum_23 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_22, %m23 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_23 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_24 = tensor.empty() : tensor<16x16xi32>
    %sum_24 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_23, %m24 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_24 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_25 = tensor.empty() : tensor<16x16xi32>
    %sum_25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_24, %m25 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_25 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_26 = tensor.empty() : tensor<16x16xi32>
    %sum_26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_25, %m26 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_26 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %sum_out_27 = tensor.empty() : tensor<16x16xi32>
    %sum_27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_26, %m27 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%sum_out_27 : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %bias = arith.constant dense<"0xEBA8084401F3020458D3FAC929270FE33FB4C0B7C65A043FECBD62E304710A8FB7622E9B7177FCB9128FFC6B0364CA388EFC5763592141214184748C620B81E8"> : tensor<16xi32>
    %biased_out = tensor.empty() : tensor<16x16xi32>
    %biased = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum_27, %bias : tensor<16x16xi32>, tensor<16xi32>) outs(%biased_out : tensor<16x16xi32>) {
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
