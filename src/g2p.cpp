#include "openv6/g2p.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>

namespace openv6::g2p {

namespace {

std::string join(const std::vector<std::string>& items, std::string_view separator, size_t from = 0) {
    std::string result;
    for (size_t i = from; i < items.size(); i++) {
        if (i > from) result += separator;
        result += items[i];
    }
    return result;
}

std::vector<char32_t> decode(std::string_view text) {
    std::vector<char32_t> result;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = text[i];
        int extra = c < 0x80 ? 0 : (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : 0;
        char32_t cp = extra == 0 ? c : c & (0x3f >> extra);
        for (int k = 1; k <= extra && i + k < text.size(); k++) cp = cp << 6 | (static_cast<unsigned char>(text[i + k]) & 0x3f);
        result.push_back(cp);
        i += extra + 1;
    }
    return result;
}

void encode(char32_t cp, std::string& out) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xc0 | cp >> 6);
        out += static_cast<char>(0x80 | (cp & 0x3f));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xe0 | cp >> 12);
        out += static_cast<char>(0x80 | (cp >> 6 & 0x3f));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    } else {
        out += static_cast<char>(0xf0 | cp >> 18);
        out += static_cast<char>(0x80 | (cp >> 12 & 0x3f));
        out += static_cast<char>(0x80 | (cp >> 6 & 0x3f));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    }
}

std::string encode(std::span<const char32_t> text) {
    std::string result;
    for (char32_t cp : text) encode(cp, result);
    return result;
}

bool is_white_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0d) || c == 0x20 || c == 0x85 || c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 ||
           c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

std::string trim_start(std::string_view text) {
    auto chars = decode(text);
    size_t first = 0;
    while (first < chars.size() && is_white_space(chars[first])) first++;
    return encode(std::span(chars).subspan(first));
}

std::string trim(std::string_view text) {
    auto chars = decode(text);
    size_t first = 0, last = chars.size();
    while (first < last && is_white_space(chars[first])) first++;
    while (last > first && is_white_space(chars[last - 1])) last--;
    return encode(std::span(chars).subspan(first, last - first));
}

bool is_blank(std::string_view text) { return trim_start(text).empty(); }

char32_t ascii_lower(char32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

std::string ascii_lower(std::string_view text) {
    std::string result(text);
    for (char& c : result) c = static_cast<char>(ascii_lower(static_cast<unsigned char>(c)));
    return result;
}

StringMap<std::string> table(std::string_view text) {
    StringMap<std::string> result;
    for (std::string_view entry : split(text, '|', true)) {
        size_t equals = entry.find('=');
        result.emplace(std::string(entry.substr(0, equals)), std::string(entry.substr(equals + 1)));
    }
    return result;
}

StringSet set_of(std::string_view text) {
    StringSet result;
    for (std::string_view item : split(text, ' ', true)) result.emplace(item);
    return result;
}

const StringSet JapaneseVowels = set_of("a i M e o");
const StringSet EnglishVowels = set_of("{ @ @r @U aI aU e eI I i: O: OI Q U u: V I@ O@ Q@ U@ e@");
const StringSet ChineseVowels = set_of("7 @N @U @_n @` AN AU UN aI a_n ei iN i_n i` y_n y a i u o ia i@U iAN iAU iE_n iE_r "
                                       "iUN u@N u@_n uAN ua uaI ua_n uei uo yE_r y{_n");

const StringMap<std::string> Kana = table(
    R"T(ぁ=a|あ=a|ぃ=i|い=i|ぅ=M|う=M|ぇ=e|え=e|ぉ=o|お=o|か=k a|が=g a|き=k' i|ぎ=g' i|く=k M|ぐ=g M|け=k e|げ=g e|こ=k o|)T"
    R"T(ご=g o|さ=s a|ざ=dz a|し=S i|じ=dZ i|す=s M|ず=dz M|せ=s e|ぜ=dz e|そ=s o|ぞ=dz o|た=t a|だ=d a|ち=tS i|ぢ=dZ i|)T"
    R"T(っ=Sil|つ=ts M|づ=dz M|て=t e|で=d e|と=t o|ど=d o|な=n a|に=J i|ぬ=n M|ね=n e|の=n o|は=h a|ば=b a|ぱ=p a|ひ=C i|)T"
    R"T(び=b' i|ぴ=p' i|ふ=p\ M|ぶ=b M|ぷ=p M|へ=h e|べ=b e|ぺ=p e|ほ=h o|ぼ=b o|ぽ=p o|ま=m a|み=m' i|む=m M|め=m e|も=m o|)T"
    R"T(ゃ=j a|や=j a|ゅ=j M|ゆ=j M|ょ=j o|よ=j o|ら=4 a|り=4' i|る=4 M|れ=4 e|ろ=4 o|ゎ=w a|わ=w a|ゐ=w i|ゑ=w e|を=o|ん=N\|)T"
    R"T(ヴ=b M|ヵ=k a|ヶ=k e|ー=-)T");

const StringMap<std::string> KanaPairs = table(
    R"T(ぃぇ=j e|いぇ=j e|ぅぃ=w i|ぅぇ=w e|ぅぉ=w o|うぃ=w i|うぇ=w e|うぉ=w o|きぇ=k' e|きゃ=k' a|きゅ=k' M|きょ=k' o|ぎぇ=g' e|)T"
    R"T(ぎゃ=g' a|ぎゅ=g' M|ぎょ=g' o|しぇ=S e|しゃ=S a|しゅ=S M|しょ=S o|じぇ=dZ e|じゃ=dZ a|じゅ=dZ M|じょ=dZ o|すぃ=s i|ずぃ=dz i|)T"
    R"T(ちぇ=tS e|ちゃ=tS a|ちゅ=tS M|ちょ=tS o|つぁ=ts a|つぃ=ts i|つぇ=ts e|つぉ=ts o|てぃ=t' i|てぇ=t' e|てゃ=t' a|てゅ=t' M|)T"
    R"T(てょ=t' o|でぃ=d' i|でぇ=d' e|でゃ=d' a|でゅ=d' M|でょ=d' o|とぅ=t M|どぅ=d M|にぇ=J e|にゃ=J a|にゅ=J M|にょ=J o|ひぇ=C e|)T"
    R"T(ひゃ=C a|ひゅ=C M|ひょ=C o|びぇ=b' e|びゃ=b' a|びゅ=b' M|びょ=b' o|ぴぇ=p' e|ぴゃ=p' a|ぴゅ=p' M|ぴょ=p' o|ふぁ=p\ a|)T"
    R"T(ふぃ=p\' i|ふぇ=p\ e|ふぉ=p\ o|ふゃ=p\' a|ふゅ=p\' M|みぇ=m' e|みゃ=m' a|みゅ=m' M|みょ=m' o|りぇ=4' e|りゃ=4' a|りゅ=4' M|)T"
    R"T(りょ=4' o)T");

const StringMap<std::string> Romaji = table(
    R"T(a=a|e=e|i=i|m=N\|n=N\|o=o|u=M|ba=b a|be=b e|bi=b' i|bo=b o|bu=b M|da=d a|de=d e|di=d' i|do=d o|)T"
    R"T(du=d M|fa=p\ a|fe=p\ e|fi=p\' i|fo=p\ o|fu=p\ M|ga=g a|ge=g e|gi=g' i|go=g o|gu=g M|ha=h a|he=h e|)T"
    R"T(hi=C i|ho=h o|ja=dZ a|je=dZ e|ji=dZ i|jo=dZ o|ju=dZ M|ka=k a|ke=k e|ki=k' i|ko=k o|ku=k M|ma=m a|)T"
    R"T(me=m e|mi=m' i|mo=m o|mu=m M|na=n a|ne=n e|ni=J i|nn=N\|no=n o|nu=n M|pa=p a|pe=p e|pi=p' i|po=p o|)T"
    R"T(pu=p M|ra=4 a|re=4 e|ri=4' i|ro=4 o|ru=4 M|sa=s a|se=s e|si=s i|so=s o|su=s M|ta=t a|te=t e|ti=t' i|)T"
    R"T(to=t o|tu=t M|wa=w a|we=w e|wi=w i|wo=w o|ya=j a|ye=j e|yo=j o|yu=j M|za=dz a|ze=dz e|zi=dz i|)T"
    R"T(zo=dz o|zu=dz M|bya=b' a|bye=b' e|byo=b' o|byu=b' M|cha=tS a|che=tS e|chi=tS i|cho=tS o|chu=tS M|)T"
    R"T(dya=d' a|dye=d' e|dyo=d' o|dyu=d' M|fya=p\' a|fyu=p\' M|gya=g' a|gye=g' e|gyo=g' o|gyu=g' M|hya=C a|)T"
    R"T(hye=C e|hyo=C o|hyu=C M|kya=k' a|kye=k' e|kyo=k' o|kyu=k' M|ltu=Sil|mya=m' a|mye=m' e|myo=m' o|)T"
    R"T(myu=m' M|nya=J a|nye=J e|nyo=J o|nyu=J M|pya=p' a|pye=p' e|pyo=p' o|pyu=p' M|rya=4' a|rye=4' e|)T"
    R"T(ryo=4' o|ryu=4' M|sha=S a|she=S e|shi=S i|sho=S o|shu=S M|tsa=ts a|tse=ts e|tsi=ts i|tso=ts o|)T"
    R"T(tsu=ts M|tya=t' a|tye=t' e|tyo=t' o|tyu=t' M|xtu=Sil|ltsu=Sil|xtsu=Sil)T");

const StringMap<std::string> Chinese = table(
    R"T(a=a|ai=aI|an=a_n|ang=AN|ao=AU|ba=p a|bai=p aI|ban=p a_n|bang=p AN|bao=p AU|bei=p ei|ben=p @_n|beng=p @N|)T"
    R"T(bi=p i|bian=p iE_n|biao=p iAU|bie=p iE_r|bin=p i_n|bing=p iN|bo=p o|bu=p u|ca=ts_h a|cai=ts_h aI|)T"
    R"T(can=ts_h a_n|cang=ts_h AN|cao=ts_h AU|ce=ts_h 7|cen=ts_h @_n|ceng=ts_h @N|cha=ts`_h a|chai=ts`_h aI|)T"
    R"T(chan=ts`_h a_n|chang=ts`_h AN|chao=ts`_h AU|che=ts`_h 7|chen=ts`_h @_n|cheng=ts`_h @N|chi=ts`_h i`|)T"
    R"T(chong=ts`_h UN|chou=ts`_h @U|chu=ts`_h u|chua=ts`_h ua|chuai=ts`_h uaI|chuan=ts`_h ua_n|chuang=ts`_h uAN|)T"
    R"T(chui=ts`_h uei|chun=ts`_h u@_n|chuo=ts`_h uo|ci=ts_h i\|cong=ts_h UN|cou=ts_h @U|cu=ts_h u|cuan=ts_h ua_n|)T"
    R"T(cui=ts_h uei|cun=ts_h u@_n|cuo=ts_h uo|da=t a|dai=t aI|dan=t a_n|dang=t AN|dao=t AU|de=t 7|dei=t ei|)T"
    R"T(den=t @_n|deng=t @N|di=t i|dia=t ia|dian=t iE_n|diao=t iAU|die=t iE_r|ding=t iN|diu=t i@U|dong=t UN|dou=t @U|)T"
    R"T(du=t u|duan=t ua_n|dui=t uei|dun=t u@_n|duo=t uo|e=7|ei=ei|en=@_n|er=@`|fa=f a|fan=f a_n|fang=f AN|fei=f ei|)T"
    R"T(fen=f @_n|feng=f @N|fo=f o|fou=f @U|fu=f u|ga=k a|gai=k aI|gan=k a_n|gang=k AN|gao=k AU|ge=k 7|gei=k ei|)T"
    R"T(gen=k @_n|geng=k @N|gong=k UN|gou=k @U|gu=k u|gua=k ua|guai=k uaI|guan=k ua_n|guang=k uAN|gui=k uei|)T"
    R"T(gun=k u@_n|guo=k uo|ha=x a|hai=x aI|han=x a_n|hang=x AN|hao=x AU|he=x 7|hei=x ei|hen=x @_n|heng=x @N|)T"
    R"T(hong=x UN|hou=x @U|hu=x u|hua=x ua|huai=x uaI|huan=x ua_n|huang=x uAN|hui=x uei|hun=x u@_n|huo=x uo|i=i|)T"
    R"T(ia=ia|ian=iE_n|iang=iAN|iao=iAU|ie=iE_r|in=i_n|ing=iN|iong=iUN|iou=i@U|iu=i@U|ji=ts\ i|jia=ts\ ia|)T"
    R"T(jian=ts\ iE_n|jiang=ts\ iAN|jiao=ts\ iAU|jie=ts\ iE_r|jin=ts\ i_n|jing=ts\ iN|jiong=ts\ iUN|jiu=ts\ i@U|)T"
    R"T(ju=ts\ y|juan=ts\ y{_n|jue=ts\ yE_r|jun=ts\ y_n|ka=k_h a|kai=k_h aI|kan=k_h a_n|kang=k_h AN|kao=k_h AU|)T"
    R"T(ke=k_h 7|kei=k_h ei|ken=k_h @_n|keng=k_h @N|kong=k_h UN|kou=k_h @U|ku=k_h u|kua=k_h ua|kuai=k_h uaI|)T"
    R"T(kuan=k_h ua_n|kuang=k_h uAN|kui=k_h uei|kun=k_h u@_n|kuo=k_h uo|la=l a|lai=l aI|lan=l a_n|lang=l AN|lao=l AU|)T"
    R"T(le=l 7|lei=l ei|leng=l @N|li=l i|lia=l ia|lian=l iE_n|liang=l iAN|liao=l iAU|lie=l iE_r|lin=l i_n|ling=l iN|)T"
    R"T(liu=l i@U|lo=l o|long=l UN|lou=l @U|lu=l u|luan=l ua_n|lun=l u@_n|luo=l uo|lv=l y|lve=l yE_r|ma=m a|mai=m aI|)T"
    R"T(man=m a_n|mang=m AN|mao=m AU|me=m 7|mei=m ei|men=m @_n|meng=m @N|mi=m i|mian=m iE_n|miao=m iAU|mie=m iE_r|)T"
    R"T(min=m i_n|ming=m iN|miu=m i@U|mo=m o|mou=m @U|mu=m u|na=n a|nai=n aI|nan=n a_n|nang=n AN|nao=n AU|ne=n 7|)T"
    R"T(nei=n ei|nen=n @_n|neng=n @N|ni=n i|nian=n iE_n|niang=n iAN|niao=n iAU|nie=n iE_r|nin=n i_n|ning=n iN|)T"
    R"T(niu=n i@U|nong=n UN|nou=n @U|nu=n u|nuan=n ua_n|nuo=n uo|nv=n y|nve=n yE_r|o=o|ou=@U|pa=p_h a|pai=p_h aI|)T"
    R"T(pan=p_h a_n|pang=p_h AN|pao=p_h AU|pei=p_h ei|pen=p_h @_n|peng=p_h @N|pi=p_h i|pian=p_h iE_n|piao=p_h iAU|)T"
    R"T(pie=p_h iE_r|pin=p_h i_n|ping=p_h iN|po=p_h o|pou=p_h @U|pu=p_h u|qi=ts\_h i|qia=ts\_h ia|qian=ts\_h iE_n|)T"
    R"T(qiang=ts\_h iAN|qiao=ts\_h iAU|qie=ts\_h iE_r|qin=ts\_h i_n|qing=ts\_h iN|qiong=ts\_h iUN|qiu=ts\_h i@U|)T"
    R"T(qu=ts\_h y|quan=ts\_h y{_n|que=ts\_h yE_r|qun=ts\_h y_n|ran=z` a_n|rang=z` AN|rao=z` AU|re=z` 7|ren=z` @_n|)T"
    R"T(reng=z` @N|ri=z` i`|rong=z` UN|rou=z` @U|ru=z` u|ruan=z` ua_n|rui=z` uei|run=z` u@_n|ruo=z` uo|sa=s a|)T"
    R"T(sai=s aI|san=s a_n|sang=s AN|sao=s AU|se=s 7|sen=s @_n|seng=s @N|sha=s` a|shai=s` aI|shan=s` a_n|shang=s` AN|)T"
    R"T(shao=s` AU|she=s` 7|shei=s` ei|shen=s` @_n|sheng=s` @N|shi=s` i`|shou=s` @U|shu=s` u|shua=s` ua|shuai=s` uaI|)T"
    R"T(shuan=s` ua_n|shuang=s` uAN|shui=s` uei|shun=s` u@_n|shuo=s` uo|si=s i\|song=s UN|sou=s @U|su=s u|)T"
    R"T(suan=s ua_n|sui=s uei|sun=s u@_n|suo=s uo|ta=t_h a|tai=t_h aI|tan=t_h a_n|tang=t_h AN|tao=t_h AU|te=t_h 7|)T"
    R"T(teng=t_h @N|ti=t_h i|tian=t_h iE_n|tiao=t_h iAU|tie=t_h iE_r|ting=t_h iN|tong=t_h UN|tou=t_h @U|tu=t_h u|)T"
    R"T(tuan=t_h ua_n|tui=t_h uei|tun=t_h u@_n|tuo=t_h uo|u=u|ua=ua|uai=uaI|uan=ua_n|uang=uAN|uei=uei|uen=u@_n|)T"
    R"T(ueng=u@N|ui=uei|un=u@_n|uo=uo|v=y|van=y{_n|ve=yE_r|vn=y_n|wa=ua|wai=uaI|wan=ua_n|wang=uAN|wei=uei|wen=u@_n|)T"
    R"T(weng=u@N|wo=uo|wu=u|xi=s\ i|xia=s\ ia|xian=s\ iE_n|xiang=s\ iAN|xiao=s\ iAU|xie=s\ iE_r|xin=s\ i_n|)T"
    R"T(xing=s\ iN|xiong=s\ iUN|xiu=s\ i@U|xu=s\ y|xuan=s\ y{_n|xue=s\ yE_r|xun=s\ y_n|ya=ia|yan=iE_n|yang=iAN|)T"
    R"T(yao=iAU|ye=iE_r|yi=i|yin=i_n|ying=iN|yong=iUN|you=i@U|yu=y|yuan=y{_n|yue=yE_r|yun=y_n|za=ts a|zai=ts aI|)T"
    R"T(zan=ts a_n|zang=ts AN|zao=ts AU|ze=ts 7|zei=ts ei|zen=ts @_n|zeng=ts @N|zha=ts` a|zhai=ts` aI|zhan=ts` a_n|)T"
    R"T(zhang=ts` AN|zhao=ts` AU|zhe=ts` 7|zhei=ts` ei|zhen=ts` @_n|zheng=ts` @N|zhi=ts` i`|zhong=ts` UN|zhou=ts` @U|)T"
    R"T(zhu=ts` u|zhua=ts` ua|zhuai=ts` uaI|zhuan=ts` ua_n|zhuang=ts` uAN|zhui=ts` uei|zhun=ts` u@_n|zhuo=ts` uo|)T"
    R"T(zi=ts i\|zong=ts UN|zou=ts @U|zu=ts u|zuan=ts ua_n|zui=ts uei|zun=ts u@_n|zuo=ts uo)T");


bool is_vowel(int language, std::string_view phoneme) {
    return (language == 0 ? JapaneseVowels : language == 1 ? EnglishVowels : ChineseVowels).contains(phoneme);
}

void move_coda(std::vector<std::string>& syllables, int start, int last, int language) {
    std::vector<std::string> tokens;
    for (auto token : split(syllables[start], ' ', false)) tokens.emplace_back(token);
    int vowel = -1;
    for (int i = static_cast<int>(tokens.size()) - 1; i >= 0; i--) {
        if (is_vowel(language, tokens[i])) {
            vowel = i;
            break;
        }
    }
    if (vowel < 0 || vowel == static_cast<int>(tokens.size()) - 1) return;
    std::vector<std::string> head(tokens.begin(), tokens.begin() + vowel + 1);
    syllables[start] = join(head, " ");
    syllables[last] = "- " + join(tokens, " ", vowel + 1);
}

std::optional<std::vector<std::string>> japanese_syllables(std::string_view lyric) {
    std::vector<char32_t> text;
    for (char32_t c : decode(lyric)) {
        c = ascii_lower(c);
        if (c == ' ' || c == 0x3000) continue;
        if (c >= U'ァ' && c <= U'ヶ' && c != U'ヴ' && c != U'ヵ' && c != U'ヶ') c -= 0x60;
        text.push_back(c);
    }
    if (text.empty()) return std::nullopt;
    std::vector<std::string> morae;
    bool kana = false, romaji = false;
    for (size_t i = 0; i < text.size();) {
        size_t length = std::min<size_t>(4, text.size() - i);
        for (; length > 0; length--) {
            std::string unit = encode(std::span(text).subspan(i, length));
            const std::string* mora = nullptr;
            static const std::string dash = "-";
            if (unit == "-") {
                mora = &dash;
            } else if (length <= 2) {
                const auto& source = length == 2 ? KanaPairs : Kana;
                if (auto it = source.find(unit); it != source.end()) {
                    mora = &it->second;
                    kana = true;
                }
            }
            if (!mora) {
                if (auto it = Romaji.find(unit); it != Romaji.end()) {
                    mora = &it->second;
                    romaji = true;
                }
            }
            if (!mora) continue;
            morae.push_back(*mora);
            i += length;
            break;
        }
        if (length == 0) return std::nullopt;
    }
    bool particle = (text[0] == U'は' || text[0] == U'へ') &&
                    std::all_of(text.begin() + 1, text.end(), [](char32_t c) { return c == U'ー'; });
    bool candidates = (romaji && !kana) || particle;
    if (candidates) return morae;
    return std::vector<std::string>{join(morae, " ")};
}

std::vector<std::string> split_english(std::string_view pronunciation) {
    std::vector<std::string> tokens;
    for (auto token : split(pronunciation, ' ', true)) tokens.emplace_back(token);
    std::vector<int> vowels;
    for (int i = 0; i < static_cast<int>(tokens.size()); i++) {
        if (EnglishVowels.contains(tokens[i])) vowels.push_back(i);
    }
    std::vector<std::string> syllables;
    int start = 0;
    for (size_t v = 0; v + 1 < vowels.size(); v++) {
        int consonants = vowels[v + 1] - vowels[v] - 1;
        int end = vowels[v] + 1 + (consonants >= 2 ? 1 : 0);
        syllables.push_back(join(std::vector<std::string>(tokens.begin() + start, tokens.begin() + end), " "));
        start = end;
    }
    syllables.push_back(join(tokens, " ", start));
    return syllables;
}

std::optional<std::vector<std::string>> english_syllables(const std::string& word, const std::string& native_directory) {
    if (word.empty()) return std::nullopt;
    auto parts = split(word, '-', true);
    std::vector<std::string> named;
    for (auto part : parts) {
        if (!part.empty()) named.emplace_back(part);
    }
    if (named.empty()) return std::vector<std::string>{"-"};
    auto pronunciation = EnglishDictionary::load(native_directory).find(join(named, "-"));
    if (!pronunciation) return std::nullopt;
    auto spoken = split_english(*pronunciation);
    if (named.size() > 1 && named.size() != spoken.size()) return std::nullopt;
    std::vector<std::string> syllables;
    size_t index = 0;
    for (auto part : parts) {
        if (!part.empty()) {
            if (named.size() == 1) syllables.insert(syllables.end(), spoken.begin(), spoken.end());
            else syllables.push_back(spoken[index]);
            index++;
        } else if (index > 0 || named.size() > 1 || spoken.size() == 1) {
            syllables.push_back("-");
        }
    }
    for (int start = 0; start < static_cast<int>(syllables.size()); start++) {
        if (syllables[start] == "-") continue;
        int last = start;
        while (last + 1 < static_cast<int>(syllables.size()) && syllables[last + 1] == "-") last++;
        if (last > start) move_coda(syllables, start, last, 1);
    }
    if (word[0] == '-') return std::vector<std::string>{join(syllables, " ")};
    return syllables;
}

std::optional<std::vector<std::string>> chinese_syllables(std::string_view lyric) {
    auto words = split(lyric, ' ', false);
    if (words.empty()) return std::nullopt;
    std::vector<std::string> syllables;
    for (std::string_view word : words) {
        size_t leading = word.find_first_not_of('-');
        if (leading == std::string_view::npos) leading = word.size();
        std::string_view body = word.substr(leading);
        size_t trailing = 0;
        while (trailing < body.size() && body[body.size() - 1 - trailing] == '-') trailing++;
        body = body.substr(0, body.size() - trailing);
        syllables.insert(syllables.end(), leading == word.size() ? word.size() : leading, "-");
        if (leading == word.size()) continue;
        auto it = Chinese.find(ascii_lower(body));
        if (it == Chinese.end()) return std::nullopt;
        syllables.push_back(it->second);
        syllables.insert(syllables.end(), trailing, "-");
    }
    if (trim_start(lyric)[0] == '-') return std::vector<std::string>{join(syllables, " ")};
    return syllables;
}

class Image {
public:
    explicit Image(std::string bytes) : bytes_(std::move(bytes)) {}

    size_t size() const { return bytes_.size(); }
    char operator[](size_t offset) const { return bytes_[offset]; }

    template <class T>
    T read(size_t offset) const {
        T value;
        std::memcpy(&value, view(offset, sizeof(T)).data(), sizeof(T));
        return value;
    }

    std::string_view view(size_t offset, size_t length) const {
        if (offset > bytes_.size() || length > bytes_.size() - offset) throw InvalidDataError("The English dictionary dll is truncated.");
        return std::string_view(bytes_).substr(offset, length);
    }

private:
    std::string bytes_;
};

}

std::vector<std::string> assign(std::string_view lyric, std::optional<std::string_view> hint, int language,
                                const std::vector<bool>& extend, const std::string& native_directory) {
    std::vector<std::string> result(extend.size(), "-");
    if (hint && !is_blank(*hint)) {
        result[0] = std::string(*hint);
        return result;
    }
    auto syllable_list = syllables(lyric, language, native_directory);
    if (!syllable_list) {
        throw InvalidDataError("V6 cannot phonemize '" + std::string(lyric) + "' in language " + std::to_string(language) + ".");
    }
    std::vector<int> syllable_notes;
    for (int i = 0; i < static_cast<int>(extend.size()); i++) {
        if (i == 0 || !extend[i]) syllable_notes.push_back(i);
    }
    for (size_t k = 0; k < syllable_notes.size(); k++) {
        if (k == syllable_notes.size() - 1) {
            std::string tail = join(*syllable_list, " ", std::min(k, syllable_list->size()));
            if (!is_blank(tail)) result[syllable_notes[k]] = tail;
        } else if (k < syllable_list->size()) {
            result[syllable_notes[k]] = (*syllable_list)[k];
        }
    }
    for (int start : syllable_notes) {
        int last = start;
        while (last + 1 < static_cast<int>(extend.size()) && extend[last + 1]) last++;
        if (last > start) move_coda(result, start, last, language);
    }
    return result;
}

std::optional<std::vector<std::string>> syllables(std::string_view lyric, int language, const std::string& native_directory) {
    switch (language) {
        case 0: return japanese_syllables(lyric);
        case 1: return english_syllables(ascii_lower(trim(lyric)), native_directory);
        case 4: return chinese_syllables(lyric);
        default: throw std::out_of_range("language");
    }
}

const EnglishDictionary& EnglishDictionary::load(const std::string& native_directory) {
    static std::mutex lock;
    static std::map<std::string, std::unique_ptr<EnglishDictionary>> loaded;
    std::string path = (std::filesystem::absolute(std::filesystem::path(native_directory) / "G2PAManager_ENG.dll")).lexically_normal().string();
    std::lock_guard guard(lock);
    auto& dictionary = loaded[path];
    if (!dictionary) dictionary.reset(new EnglishDictionary(path));
    return *dictionary;
}

std::optional<std::string> EnglishDictionary::find(std::string_view word) const {
    const auto& words = word.find('-') != std::string_view::npos ? hyphenated_ : plain_;
    auto it = words.find(word);
    if (it == words.end()) return std::nullopt;
    return it->second;
}

EnglishDictionary::EnglishDictionary(const std::string& path) {
    static const StringSet phonemes = [] {
        StringSet set = EnglishVowels;
        for (auto p : split("b bh d dh D dZ f g gh h j k kh l l0 m n N p ph r s S t th T tS v w z Z", ' ', true)) set.emplace(p);
        return set;
    }();
    const Image dll(read_text(path));
    struct Section {
        uint32_t address, size, offset;
    };
    size_t pe = dll.read<uint32_t>(0x3c);
    size_t optional = pe + 24;
    uint64_t image_base = dll.read<uint64_t>(optional + 24);
    size_t section_table = optional + dll.read<uint16_t>(pe + 20);
    std::map<std::string, Section> sections;
    for (int i = 0, count = dll.read<uint16_t>(pe + 6); i < count; i++) {
        size_t header = section_table + 40 * i;
        std::string name(dll.view(header, 8));
        name.resize(name.find('\0') == std::string::npos ? name.size() : name.find('\0'));
        sections[name] = {dll.read<uint32_t>(header + 12), dll.read<uint32_t>(header + 16), dll.read<uint32_t>(header + 20)};
    }
    if (!sections.contains(".rdata") || !sections.contains(".data")) {
        throw InvalidDataError(path + " holds no VOCALOID6 English dictionary.");
    }
    const Section rdata = sections[".rdata"], data = sections[".data"];
    auto text = [&](uint64_t pointer) -> std::optional<std::string> {
        if (pointer < image_base + rdata.address) return std::nullopt;
        uint64_t relative = pointer - image_base - rdata.address;
        if (relative >= rdata.size) return std::nullopt;
        size_t start = rdata.offset + relative;
        std::string result;
        for (size_t end = start; end + 1 < dll.size() && (dll[end] | dll[end + 1]) != 0; end += 2) {
            result += dll[end + 1] == 0 && static_cast<unsigned char>(dll[end]) < 0x80 ? dll[end] : '?';
        }
        return result;
    };
    for (size_t o = data.offset; o + 16 <= static_cast<size_t>(data.offset) + data.size; o += 16) {
        auto word = text(dll.read<uint64_t>(o));
        auto pronunciation = text(dll.read<uint64_t>(o + 8));
        if (!word || !pronunciation || word->empty() || pronunciation->empty()) continue;
        bool valid = std::all_of(word->begin(), word->end(), [](char c) { return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '\'' || c == '-'; });
        for (auto token : split(*pronunciation, ' ', true)) valid = valid && phonemes.contains(token);
        if (!valid) continue;
        std::string lower = ascii_lower(*word), unhyphenated;
        for (char c : lower) {
            if (c != '-') unhyphenated += c;
        }
        plain_.emplace(unhyphenated, *pronunciation);
        hyphenated_.emplace(lower, *pronunciation);
    }
    if (plain_.empty()) throw InvalidDataError(path + " holds no VOCALOID6 English dictionary.");
}

}
