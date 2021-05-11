import urllib.request
import bs4


def get_all_urls(data):
    clean_data = str(data).split()
    print(clean_data)
    all = []
    for piece in clean_data:
        if 'href="' in piece and '/ru/baccalaureate' in piece and 'https' in piece:
            all.append(piece[6:piece.find('">')])
    return all


def open_html():
    url = "https://priem-rtf.urfu.ru/ru/baccalaureate/"
    kek = urllib.request.urlopen(url)
    content = kek.read()
    print(content)
    refs = get_all_urls(content)
    programms = {}
    # print('\n'.join(refs))
    for ref in refs[:-1]:
        print()
        print(ref)
        kek = urllib.request.urlopen(ref)
        content = kek.read()
        # print(content)
        soup = bs4.BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        begin = text.find('1 КУРС')
        print(text)
        programms[ref] = text[begin:text.rfind('Полный учебный план')]
    return programms


def main():
    open_html()


if __name__ == "__main__":
    main()